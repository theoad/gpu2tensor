import io
import json
import zipfile

import numpy as np
import pytest
import torch

from gpu2tensor import Candidate, Evaluator, Workload
from gpu2tensor.client import pack_request, unpack_result
from gpu2tensor.runner import evaluate_archive
from gpu2tensor.tensors import decode_storage, input_storage, logical_dtype


def test_all_bf16_bit_patterns_survive_wire_transport():
    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    tensor = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
    target = Workload("def reference(x):\n    return x\n", [(tensor,)])
    payload = pack_request(Candidate("def run(x):\n    return x\n", "python"), target, False, 1)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read("request.json"))
        assert manifest["version"] == 2 and manifest["input_dtypes"] == [["bfloat16"]]
        saved = np.load(io.BytesIO(archive.read("inputs/0-0.npy")), allow_pickle=False)
    restored = decode_storage(saved, "bfloat16")
    assert logical_dtype(restored) == "bfloat16"
    np.testing.assert_array_equal(input_storage(restored)[0], bits)


def test_bf16_matmul_and_exact_output_artifacts():
    a = torch.tensor([[1, 2, 4], [-1, 8, 2]], dtype=torch.bfloat16)
    b = torch.tensor([[1, 2], [2, -1], [4, 1]], dtype=torch.bfloat16)
    reference = "def reference(a, b):\n    return (a.double() @ b.double()).to(a.dtype)\n"
    target = Workload(reference, [(a, b)], capture_outputs=True)
    source = "def run(a, b):\n    return (a.float() @ b.float()).to(a.dtype)\n"
    result = Evaluator(backend="cpu").evaluate(Candidate(source, "python"), target, repetitions=2)
    assert result.correct and result.record["inputs"][0][0]["dtype"] == "bfloat16"
    descriptor = result.record["outputs"][0]
    assert descriptor["dtype"] == "bfloat16" and descriptor["storage_dtype"] == "uint16"
    saved = np.load(io.BytesIO(result.artifacts[descriptor["file"]]), allow_pickle=False)
    torch.testing.assert_close(decode_storage(saved, "bfloat16"), (a.float() @ b.float()).bfloat16(), rtol=0, atol=0)


def test_validator_cannot_silently_accept_wrong_logical_dtype():
    target = Workload("def reference(x):\n    return x\n", [(torch.ones(2, dtype=torch.bfloat16),)],
                      validator="def validate(inputs, actual):\n    return {'passed': True}\n")
    result = Evaluator(backend="cpu").evaluate(Candidate("def run(x):\n    return x.float()\n", "python"), target)
    assert result.record["status"] == "incorrect"
    assert "dtype" in result.record["message"] and result.latency_ms is None


def test_workload_validator_rejects_before_timing_and_retains_failed_output():
    target = Workload("def reference(x):\n    return x\n", [(torch.ones(2, dtype=torch.bfloat16),)],
                      atol=100, validator="""import torch
def validate(inputs, actual):
    return {'passed': torch.equal(inputs[0], actual), 'reason': 'Exact check failed', 'metrics': {'checked': actual.numel()}}
""", capture_outputs=True)
    result = Evaluator(backend="cpu").evaluate(Candidate("def run(x):\n    return x + 1\n", "python"), target)
    assert result.record["status"] == "incorrect" and result.latency_ms is None
    assert result.record["validation"][0]["metrics"]["checked"] == 2
    assert result.record["message"] == "Exact check failed"
    assert "outputs/0.npy" in result.artifacts and "validator.py" in result.artifacts


def test_custom_validator_replaces_global_tolerance():
    target = Workload("def reference(x):\n    return x\n", [(np.ones(2, dtype=np.float32),)], rtol=0, atol=0,
                      validator="def validate(inputs, actual):\n    return {'passed': bool(((actual-inputs[0]).abs() <= 1).all()), 'metrics': {}}\n")
    result = Evaluator(backend="cpu").evaluate(Candidate("def run(x):\n    return x + 0.5\n", "python"), target, repetitions=1)
    assert result.correct and result.record["validation_method"] == "workload_validator"


def test_old_float_request_still_runs():
    target = Workload("def reference(x):\n    return x\n", [(np.ones(2, dtype=np.float32),)])
    payload = pack_request(Candidate("def run(x):\n    return x\n", "python"), target, False, 1)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(buffer, "w") as old:
        for name in original.namelist():
            data = original.read(name)
            if name == "request.json":
                manifest = json.loads(data)
                manifest["version"] = 1
                manifest.pop("input_dtypes")
                data = json.dumps(manifest).encode()
            old.writestr(name, data)
    result = unpack_result(evaluate_archive(buffer.getvalue(), "cpu"))
    assert result.correct and result.record["version"] == 1


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64, np.uint64, np.bool_])
def test_numpy_storage_and_logical_dtype_are_preserved(dtype):
    values = np.array([0, 1], dtype=dtype)
    storage, logical = input_storage(values)
    assert storage is values
    np.testing.assert_array_equal(decode_storage(storage, logical), values)
