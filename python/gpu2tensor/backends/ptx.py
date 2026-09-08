"""Assemble PTX and launch its explicit ABI through the CUDA driver."""

import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from gpu2tensor.ptx import Input, Launch, Output, Scalar, Workspace


class AssemblyError(ValueError):
    """ptxas reported a source diagnostic, rather than a process failure."""


def assemble(source, output, architecture, *, assembler=None):
    """CPU-only assembly is useful for qualification when a GPU is unavailable."""
    if assembler is None:
        assembler = shutil.which("ptxas")
        if assembler is None:
            import triton
            assembler = str(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ptx, cubin = output / "candidate.ptx", output / "candidate.cubin"
    ptx.write_text(source)
    command = [str(assembler), "-arch", architecture, "-v", str(ptx), "-o", str(cubin)]
    version = subprocess.check_output([str(assembler), "--version"], text=True, timeout=10).strip()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output / "ptxas.log").write_text(result.stdout + result.stderr)
    metadata = {"assembler": str(assembler), "version": version, "command": command,
                "architecture": architecture, "returncode": result.returncode,
                "source_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest()}
    if result.returncode == 0:
        metadata["cubin_sha256"] = hashlib.sha256(cubin.read_bytes()).hexdigest()
    (output / "assembly.json").write_text(json.dumps(metadata, indent=2))
    if result.returncode:
        # Configuration/tool failures must not become invalid-program labels.
        if result.returncode > 0 and "error" in result.stderr.lower() and "line " in result.stderr.lower():
            raise AssemblyError(result.stderr[-4000:])
        raise RuntimeError(result.stderr[-4000:])
    return cubin, metadata


class Driver:
    def __init__(self):
        self.library = ct.CDLL("libcuda.so.1")
        pointer = ct.c_void_p
        self.bind("cuModuleLoad", [ct.POINTER(pointer), ct.c_char_p])
        self.bind("cuModuleGetFunction", [ct.POINTER(pointer), pointer, ct.c_char_p])
        self.bind("cuFuncGetAttribute", [ct.POINTER(ct.c_int), ct.c_int, pointer])
        self.bind("cuFuncSetAttribute", [pointer, ct.c_int, ct.c_int])
        self.bind("cuLaunchKernel", [pointer] + [ct.c_uint] * 7 + [pointer, ct.POINTER(pointer), ct.POINTER(pointer)])

    def bind(self, name, arguments):
        function = getattr(self.library, name)
        function.argtypes = arguments
        function.restype = ct.c_int

    def call(self, name, *arguments):
        code = getattr(self.library, name)(*arguments)
        if code:
            raise RuntimeError(f"CUDA driver {name} failed with code {code}.")


SCALARS = {"int32": ct.c_int32, "uint32": ct.c_uint32, "int64": ct.c_int64,
           "uint64": ct.c_uint64, "float32": ct.c_float, "float64": ct.c_double}


class Program:
    def __init__(self, source, launch, output):
        import torch
        self.launch = Launch.from_dict(launch)
        self.output_poison = None
        torch.cuda.init()
        # Torch owns the primary context, allocator, stream, and tensor lifetimes.
        torch.empty(0, device="cuda")
        major, minor = torch.cuda.get_device_capability()
        cubin, self.assembly = assemble(source, output, f"sm_{major}{minor}")
        self.driver = Driver()
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        self.driver.call("cuModuleLoad", ct.byref(self.module), str(cubin).encode())
        self.driver.call("cuModuleGetFunction", ct.byref(self.function), self.module, self.launch.entry.encode())
        names = ("max_threads_per_block", "static_shared_bytes", "constant_bytes", "local_bytes_per_thread",
                 "registers_per_thread", "ptx_version", "binary_version")
        self.resources = {}
        for attribute, name in enumerate(names):
            value = ct.c_int()
            self.driver.call("cuFuncGetAttribute", ct.byref(value), attribute, self.function)
            self.resources[name] = value.value
        if self.launch.shared_memory_bytes > 48 * 1024:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
            self.driver.call("cuFuncSetAttribute", self.function, 8, self.launch.shared_memory_bytes)
        (output / "launch.json").write_text(json.dumps(self.launch.to_dict(), indent=2))
        (output / "resources.json").write_text(json.dumps(self.resources, indent=2))

    def run(self, *inputs):
        import torch
        values, owned, output = [], [], None
        for argument in self.launch.arguments:
            if isinstance(argument, Scalar):
                values.append(SCALARS[argument.dtype](argument.value))
                continue
            if isinstance(argument, Input):
                value = inputs[argument.index]
            else:
                allocator = torch.zeros if isinstance(argument, Workspace) and argument.initialization == "zero" else torch.empty
                value = allocator(argument.shape, dtype=getattr(torch, argument.dtype), device="cuda")
                owned.append(value)
                if isinstance(argument, Output):
                    output = value
                    if self.output_poison is not None:
                        value.view(torch.uint8).fill_(self.output_poison)
            if value.data_ptr() % argument.alignment:
                raise ValueError("A buffer does not satisfy its declared alignment.")
            values.append(ct.c_uint64(value.data_ptr()))
        pointers = (ct.c_void_p * len(values))(*(ct.addressof(value) for value in values))
        stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.driver.call("cuLaunchKernel", self.function, *self.launch.grid, *self.launch.block,
                         self.launch.shared_memory_bytes, stream, pointers, None)
        # All allocations use this stream, so Torch can safely defer their reuse.
        return output


def prepare(source, launch, output):
    from gpu2tensor.backends.cuda import Runner
    program = Program(source, launch, output)
    runner = Runner(program)
    runner.metadata = {"ptx_launch": program.launch.to_dict(), "assembly": program.assembly,
                       "resources": program.resources, "cuda_runtime_qualification": "pending"}
    return runner
