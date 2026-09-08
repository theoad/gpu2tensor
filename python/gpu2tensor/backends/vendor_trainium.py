"""Official Torch-Neuron compilation with the NKI adapter's resident NRT timer."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from gpu2tensor.backends.trainium import Runner
from gpu2tensor.tensors import input_storage


class Program:
    def __init__(self, metadata, case, names):
        self.metadata = metadata
        self.neff_path = metadata["neff"]
        self.case, self.names = case, names
        if len(metadata["inputs"]) != len(case) or len(metadata["outputs"]) != 1:
            raise ValueError("Vendor baseline must compile to the declared inputs and one output.")
        for value, spec in zip(case, metadata["inputs"]):
            if list(value.shape) != spec["shape"] or spec["dtype"] != "BFLOAT16":
                raise ValueError("Vendor compiler input layout differs from the runtime contract.")
        self.output_specs = metadata["outputs"]

    def prepare_inputs(self, values):
        return {spec["name"]: values[name] for spec, name in zip(self.metadata["inputs"], self.names)}

    def prepare_outputs(self):
        spec = self.output_specs[0]
        # This first vendor baseline explicitly targets BF16 inputs and output.
        if spec["dtype"] != "BFLOAT16":
            raise ValueError("The vendor baseline contract requires BF16 output.")
        import ml_dtypes
        return {spec["name"]: np.empty(spec["shape"], dtype=ml_dtypes.bfloat16)}


class VendorRunner(Runner):
    def __init__(self, module):
        super().__init__(SimpleNamespace(kernel=module.run))
        self.source = Path(module.__file__).read_text()
        self.compiler_python = os.environ.get("GPU2TENSOR_TORCH_NEURON_PYTHON")
        if not self.compiler_python:
            raise RuntimeError("Set GPU2TENSOR_TORCH_NEURON_PYTHON to the operator's qualified compiler environment.")

    def identity(self):
        from gpu2tensor.vendor import matmul
        baseline = "torch_neuronx_matmul" if self.source == matmul("trainium").source else None
        return {**super().identity(), "vendor_baseline": baseline,
                "vendor_compiler_python": self.compiler_python}

    def compile(self, case):
        if any(value.dtype.name != "bfloat16" for value in case):
            raise ValueError("The vendor baseline currently expects BF16 inputs.")
        key = tuple((value.shape, str(value.dtype)) for value in case)
        if key not in self.compiled:
            directory = Path(self.directory.name) / str(len(self.compiled))
            directory.mkdir()
            (directory / "source.py").write_text(self.source)
            dtypes = []
            for index, value in enumerate(case):
                storage, dtype = input_storage(value)
                np.save(directory / f"input-{index}.npy", storage, allow_pickle=False)
                dtypes.append(dtype)
            (directory / "compile.json").write_text(json.dumps({"dtypes": dtypes}))
            environment = os.environ.copy()
            environment["PATH"] = str(Path(self.compiler_python).parent) + os.pathsep + environment.get("PATH", "")
            with (directory / "compile.log").open("w") as log:
                result = subprocess.run([self.compiler_python, "-m", "gpu2tensor.backends.vendor_compile", str(directory)],
                                        cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if result.returncode:
                raise RuntimeError((directory / "compile.log").read_text()[-6000:])
            metadata = json.loads((directory / "compiled.json").read_text())
            self.compiled[key] = Program(metadata, case, self.parameter_names)
        return self.compiled[key]

    def save_artifacts(self, output):
        import shutil
        result = super().save_artifacts(output)
        for source in sorted(Path(self.directory.name).iterdir()):
            index = source.name
            for name in ("metaneff.pb", "compiled.json", "compile.log", "compiler/command.txt", "compiler/model/graph.hlo"):
                path = source / name
                if path.is_file():
                    destination = output / f"vendor/{index}" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
        return result


def prepare(module):
    return VendorRunner(module)
