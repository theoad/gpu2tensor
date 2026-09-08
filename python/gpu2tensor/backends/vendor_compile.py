"""Compile Torch to a NEFF in an operator-provided Torch-Neuron environment."""

import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    import numpy as np
    import torch
    import torch_neuronx
    from torch_neuronx.proto import metaneff_pb2
    from torch_neuronx.xla_impl.trace import _trace
    from gpu2tensor.execute import load_module
    from gpu2tensor.tensors import decode_storage, torch_input

    directory = Path(sys.argv[1])
    version = importlib.metadata.version("torch-neuronx")
    if not version.startswith("2.9."):
        raise RuntimeError("Qualify this NEFF extraction adapter for the installed Torch-Neuron version.")
    request = json.loads((directory / "compile.json").read_text())
    module = load_module(directory / "source.py", "gpu2tensor_vendor_source")
    inputs = tuple(torch_input(decode_storage(np.load(directory / f"input-{index}.npy", allow_pickle=False), dtype))
                   for index, dtype in enumerate(request["dtypes"]))
    # The vendor trace frontend supplies ABI metadata before its TorchScript
    # runtime wrapper. Our measured execution uses the same NRT path as NKI.
    neff, metadata, _, _, weights = _trace(module.run, inputs, compiler_workdir=str(directory / "compiler"),
                                          compiler_args=["--target=trn1", "--auto-cast=none"], cpu_backend=True)
    if weights:
        raise ValueError("The baseline adapter expects runtime inputs and no captured weights.")
    description = metaneff_pb2.MetaNeff()
    description.ParseFromString(metadata)
    (directory / "metaneff.pb").write_bytes(metadata)
    def tensor(value):
        return {"name": value.name.decode(), "shape": list(value.shape),
                "dtype": metaneff_pb2.MetaTensor.DataType.Name(value.data_type)}
    result = {"neff": neff, "inputs": [tensor(v) for v in description.input_tensors],
              "outputs": [tensor(v) for v in description.output_tensors],
              "torch_neuronx": version, "torch": torch.__version__,
              "neuronx_cc": importlib.metadata.version("neuronx-cc"),
              "compiler_args": ["--target=trn1", "--auto-cast=none"], "frontend": "torch_neuronx_trace",
              "dependencies": {name: importlib.metadata.version(name) for name in ("islpy", "numpy", "protobuf", "torch-xla", "libneuronxla")},
              "precision": {"inputs": "bfloat16", "outputs": "bfloat16", "auto_cast": "none", "accumulation": "vendor_default"}}
    (directory / "compiled.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
