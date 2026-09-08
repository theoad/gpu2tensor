"""Child-process entry point. Vendor imports belong here and in backends."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def host_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate(directory, backend):
    import torch

    request = json.loads((directory / "request.json").read_text())
    if request["version"] != 1:
        raise ValueError("Unsupported request version.")
    if not 1 <= request["repetitions"] <= 10000:
        raise ValueError("Invalid repetition count.")
    if backend == "cuda" and request["language"] != "triton":
        raise ValueError("The CUDA v0 adapter accepts Triton Python modules.")
    if backend == "trainium" and request["language"] != "nki":
        raise ValueError("The Trainium v0 adapter accepts NKI Python modules.")
    output = directory / "output"
    record = {"version": 1, "backend": backend, "language": request["language"],
              "name": request["name"], "status": "error", "profile_status": "not_requested",
              "source_sha256": hashlib.sha256((directory / "candidate.py").read_bytes()).hexdigest(),
              "request_sha256": hashlib.sha256((directory / "request.json").read_bytes()).hexdigest(),
              "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
              "kernel": platform.release(), "ami": os.environ.get("GPU2TENSOR_AMI"),
              "benchmark_case": 0, "correctness_cases": 0, "max_absolute_error": 0.0}
    stage = "load"
    try:
        reference = load_module(directory / "reference.py", "gpu2tensor_reference")
        module = load_module(directory / "candidate.py", "gpu2tensor_candidate")
        cases = [tuple(np.load(directory / f"inputs/{case_index}-{index}.npy", allow_pickle=False)
                       for index in range(size)) for case_index, size in enumerate(request["case_sizes"])]
        if not cases or not cases[0]:
            raise ValueError("No input cases.")
        record["inputs"] = [[{"shape": list(value.shape), "dtype": str(value.dtype),
                              "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                             for value in case] for case in cases]
        torch.set_num_threads(1)
        stage = "prepare"
        if backend == "cuda":
            from gpu2tensor.backends.cuda import prepare
        elif backend == "trainium":
            from gpu2tensor.backends.trainium import prepare
        else:
            from gpu2tensor.backends.cpu import prepare
        runner = prepare(module)
        record.update(runner.identity())
        backend_source = Path(sys.modules[type(runner).__module__].__file__).read_bytes()
        record["backend_sha256"] = hashlib.sha256(backend_source).hexdigest()
        (output / "backend.py").write_bytes(backend_source)
        stage = "correctness"
        started = time.monotonic()
        for case in cases:
            # Independent copies keep input mutation out of the reference result.
            expected = host_array(reference.reference(*(torch.from_numpy(value.copy()) for value in case)))
            actual = host_array(runner.run(tuple(value.copy() for value in case)))
            if actual.shape != expected.shape:
                record.update(status="incorrect", message=f"Output shape {actual.shape} differs from {expected.shape}.")
                return record
            if actual.dtype != expected.dtype:
                record.update(status="incorrect", message=f"Output dtype {actual.dtype} differs from {expected.dtype}.")
                return record
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                record.update(status="incorrect", message="This finite-input contract requires finite outputs.")
                return record
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            record["max_absolute_error"] = max(error, record["max_absolute_error"])
            if not np.allclose(actual, expected, rtol=request["rtol"], atol=request["atol"], equal_nan=False):
                record.update(status="incorrect", message="Output differs from the workload reference.")
                return record
            record["correctness_cases"] += 1
        record["prepare_and_check_seconds"] = time.monotonic() - started
        stage = "benchmark"
        record["latency_ms"] = runner.benchmark(cases[0], request["repetitions"])
        record["timing_method"] = runner.timing_method
        record["status"] = "ok"
        if request["profile"]:
            stage = "profile"
            profile_started = time.monotonic()
            try:
                record["profile"] = runner.profile(cases[0], output)
                record["profile_status"] = "ok"
            except Exception as error:
                record["profile_status"] = "error"
                record["profile_error"] = str(error)
                traceback.print_exc()
            finally:
                record["profile_seconds"] = time.monotonic() - profile_started
        return record
    except Exception as error:
        record.update(status="error", stage=stage, message=str(error))
        traceback.print_exc()
        return record


def main():
    directory = Path(sys.argv[1])
    record = evaluate(directory, sys.argv[2])
    (directory / "output/record.json").write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
