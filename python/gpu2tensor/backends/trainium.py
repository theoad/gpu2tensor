"""NKI compilation and Neuron runtime execution, with separate NTFF capture."""

import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


class Runner:
    timing_method = "neuron_host_round_trip_with_resident_tensors"

    def __init__(self, module):
        import nki
        from nki.compiler.frontend import ParserFrontend
        from nki.compiler.ncc_driver import CompileOptions
        from nki.framework.compiled import compile_kernel_to_nir, run_from_bir
        from nki.runtime import SpikeModel, SpikeTensor

        if os.environ.get("NKI_SIMULATOR") == "1":
            raise RuntimeError("Unset NKI_SIMULATOR for hardware evaluation.")
        self.module = module
        self.nki = nki
        if not nki.__version__.startswith("0.6."):
            raise RuntimeError("This adapter is qualified for NKI 0.6; qualify other versions before use.")
        self.frontend = ParserFrontend
        self.compile_ir = compile_kernel_to_nir
        self.compile_binary = run_from_bir
        self.CompileOptions = CompileOptions
        self.Model = SpikeModel
        self.Tensor = SpikeTensor
        self.target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
        visible = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
        if not visible.isdecimal():
            raise ValueError("Assign one physical core with NEURON_RT_VISIBLE_CORES.")
        self.physical_core = int(visible)
        # Neuron remaps the single visible physical core to logical core zero.
        self.core_id = 0
        self.compiled = {}
        self.directory = tempfile.TemporaryDirectory(prefix="gpu2tensor-neuron-")
        self.parameter_names = tuple(inspect.signature(module.kernel).parameters)

    def identity(self):
        return {"device": self.target, "neuron_core": self.physical_core,
                "runtime_neuron_core": self.core_id, "nki": self.nki.__version__,
                "neuronx_cc": importlib.metadata.version("neuronx-cc"),
                "neuron_devices": subprocess.check_output(["/opt/aws/neuron/bin/neuron-ls", "--json-output"], text=True)}

    def bind(self, case):
        if len(case) != len(self.parameter_names):
            raise ValueError("The input count differs from the kernel signature.")
        return dict(zip(self.parameter_names, case))

    def compile(self, case):
        key = tuple((value.shape, str(value.dtype)) for value in case)
        if key not in self.compiled:
            directory = Path(self.directory.name) / str(len(self.compiled))
            directory.mkdir()
            options = self.CompileOptions(target=self.target, artifacts_dir=str(directory),
                                          output_path=str(directory / "kernel.neff"))
            inputs = self.bind(case)
            ir = self.compile_ir(self.nki.jit(self.module.kernel), inputs=inputs,
                                 compile_opts=options, frontend=self.frontend(), enable_cache=False)
            compiled_results = []

            def keep_compiled_program(compiled, inputs, outputs):
                # NKI exposes binary compilation through an execution pipeline.
                # Capture its compiled object; correctness owns the first run.
                compiled_results.append(compiled)
                for value in outputs.values():
                    value.fill(0)

            self.compile_binary(ir, options, inputs, executor=keep_compiled_program)
            compiled = compiled_results[0]
            if len(compiled.output_specs) != 1:
                raise ValueError("The v0 workload contract expects one output tensor.")
            self.compiled[key] = compiled
        return self.compiled[key]

    def run(self, case):
        model, inputs, outputs = self.resident(case)
        model(inputs, outputs=outputs, save_trace=False)
        return next(iter(outputs.values())).numpy()

    def resident(self, case):
        compiled = self.compile(case)
        model = self.Model.load_from_neff(neff_path=compiled.neff_path, core_id=self.core_id)
        inputs = {name: self.Tensor.from_numpy(value.copy(), name=name, core_id=self.core_id)
                  for name, value in compiled.prepare_inputs(self.bind(case)).items()}
        outputs = {name: self.Tensor.from_numpy(value, name=name, core_id=self.core_id)
                   for name, value in compiled.prepare_outputs().items()}
        return model, inputs, outputs

    def benchmark(self, case, repetitions):
        model, inputs, outputs = self.resident(case)
        for _ in range(5):
            model(inputs, outputs=outputs, save_trace=False)
        samples = []
        for _ in range(repetitions):
            # Restore resident inputs before timing so mutations cannot accumulate.
            compiled = self.compile(case)
            inputs = {name: self.Tensor.from_numpy(value.copy(), name=name, core_id=self.core_id)
                      for name, value in compiled.prepare_inputs(self.bind(case)).items()}
            start = time.perf_counter_ns()
            model(inputs, outputs=outputs, save_trace=False)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return samples

    def profile(self, case, output):
        compiled = self.compile(case)
        model, inputs, outputs = self.resident(case)
        model(inputs, outputs=outputs, save_trace=True, ntff_name=str(output / "kernel.ntff"))
        shutil.copyfile(compiled.neff_path, output / "kernel.neff")
        if not (output / "kernel.ntff").is_file():
            raise RuntimeError("Neuron runtime did not produce the requested NTFF trace.")
        profile = {"provider": "neuron_runtime", "collection": "separate_execution",
                "scope": "device_instruction_trace", "completeness": "not_verified",
                "replay": "not_requested", "sampling": "not_reported", "dropped_events": None,
                "artifacts": ["kernel.neff", "kernel.ntff"]}
        explorer = shutil.which("neuron-explorer")
        if explorer is None:
            profile["summary_status"] = "tool_unavailable"
            return profile
        # Preserve the native summary. Expose only counters whose units are
        # explicit; vendor engine times and utilization need separate semantics.
        parsed = subprocess.run(
            [explorer, "view", "-n", str(output / "kernel.neff"),
             "-s", str(output / "kernel.ntff"), "--output-format", "summary-json",
             "--disable-ui", "--ingest-only"], capture_output=True, text=True, timeout=60,
        )
        (output / "explorer.log").write_text(parsed.stderr)
        profile["artifacts"].append("explorer.log")
        if parsed.returncode:
            profile.update(summary_status="error", summary_error=parsed.stderr[-2000:])
            return profile
        summary = json.loads(parsed.stdout)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        profile["artifacts"].append("summary.json")
        if len(summary) != 1:
            profile["summary_status"] = "multiple_sessions_not_supported"
            return profile
        session = next(iter(summary.values()))
        counts = ("event_count", "trace_count", "neuroncore_cycle_count", "hbm_read_bytes",
                  "hbm_write_bytes", "sbuf_read_bytes", "sbuf_write_bytes")
        counts += tuple(f"{engine}_engine_instruction_count" for engine in
                        ("scalar", "vector", "tensor", "gpsimd", "sync"))
        profile["measurements"] = {key: session[key] for key in counts if key in session}
        profile["summary_status"] = "ok"
        profile["explorer_version"] = session.get("profiler_version")
        profile["runtime_version"] = session.get("runtime_version")
        return profile


def prepare(module):
    return Runner(module)
