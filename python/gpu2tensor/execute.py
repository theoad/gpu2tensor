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

from gpu2tensor.tensors import copy_input, decode_storage, host_array, input_storage, logical_dtype, torch_input


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reject(record, case, message):
    record.update(status="incorrect", failed_case=case, failure_kind="validation", message=message)
    record["validation"].append({"case": case, "passed": False, "reason": message, "metrics": {}})


def evaluate(directory, backend):
    import torch

    request = json.loads((directory / "request.json").read_text())
    if request["version"] not in (1, 2):
        raise ValueError("Unsupported request version.")
    if not 1 <= request["repetitions"] <= 10000:
        raise ValueError("Invalid repetition count.")
    if request["language"] == "ptx" and backend != "cuda":
        raise ValueError("PTX requires a CUDA worker.")
    if backend == "cuda" and request["language"] not in ("triton", "ptx", "torch"):
        raise ValueError("The CUDA adapter accepts Triton, PTX, or Torch candidates.")
    if backend == "trainium" and request["language"] != "nki":
        raise ValueError("The Trainium v0 adapter accepts NKI Python modules.")
    output = directory / "output"
    record = {"version": request["version"], "backend": backend, "language": request["language"],
              "name": request["name"], "status": "error", "profile_status": "not_requested",
              "source_sha256": hashlib.sha256((directory / "candidate.py").read_bytes()).hexdigest(),
              "request_sha256": hashlib.sha256((directory / "request.json").read_bytes()).hexdigest(),
              "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
              "kernel": platform.release(), "ami": os.environ.get("GPU2TENSOR_AMI"),
              "benchmark_case": 0, "correctness_cases": 0, "max_absolute_error": 0.0}
    stage = "load"
    try:
        package = Path(__file__).parent
        engine = output / "engine"
        record["engine_sha256"] = {}
        for path in list(package.glob("*.py")) + list((package / "backends").glob("*.py")):
            relative = path.relative_to(package)
            source = path.read_bytes()
            target = engine / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source)
            record["engine_sha256"][str(relative)] = hashlib.sha256(source).hexdigest()
        reference = load_module(directory / "reference.py", "gpu2tensor_reference")
        validator = load_module(directory / "validator.py", "gpu2tensor_validator") if request.get("validator") else None
        module = None if request["language"] == "ptx" else load_module(directory / "candidate.py", "gpu2tensor_candidate")
        cases = []
        for case_index, size in enumerate(request["case_sizes"]):
            case = []
            for index in range(size):
                storage = np.load(directory / f"inputs/{case_index}-{index}.npy", allow_pickle=False)
                dtype = request["input_dtypes"][case_index][index] if request["version"] == 2 else storage.dtype.name
                case.append(decode_storage(storage, dtype))
            cases.append(tuple(case))
        if not cases or not cases[0]:
            raise ValueError("No input cases.")
        record["inputs"] = [[{"shape": list(value.shape), "dtype": logical_dtype(value),
                              "sha256": hashlib.sha256(input_storage(value)[0].tobytes()).hexdigest()}
                             for value in case] for case in cases]
        record["validation_method"] = "workload_validator" if validator else "allclose"
        record["validation"] = []
        record["diagnostics"] = []
        if request.get("capture_outputs"):
            record["outputs"] = []
        torch.set_num_threads(1)
        stage = "prepare"
        if backend == "cuda":
            from gpu2tensor.backends.cuda import prepare
        elif backend == "trainium":
            from gpu2tensor.backends.trainium import prepare
        else:
            from gpu2tensor.backends.cpu import prepare
        if request["language"] == "ptx":
            from gpu2tensor.backends.ptx import prepare as prepare_ptx
            runner = prepare_ptx((directory / "candidate.py").read_text(), request["launch"], output)
        else:
            runner = prepare(module)
        record.update(runner.identity())
        backend_source = Path(sys.modules[type(runner).__module__].__file__).read_bytes()
        record["backend_sha256"] = hashlib.sha256(backend_source).hexdigest()
        (output / "backend.py").write_bytes(backend_source)
        stage = "correctness"
        started = time.monotonic()
        for case_index, case in enumerate(cases):
            stage = "correctness"
            # Independent copies keep input mutation out of the reference result.
            expected_value = reference.reference(*(torch_input(copy_input(value)) for value in case))
            actual_value = runner.run(tuple(copy_input(value) for value in case))
            if hasattr(runner, "save_artifacts"):
                record["programs"] = runner.save_artifacts(output)
            expected_tensor = torch_input(expected_value).detach().cpu()
            actual_tensor = torch_input(actual_value).detach().cpu()
            expected_dtype, actual_dtype = logical_dtype(expected_tensor), logical_dtype(actual_tensor)
            if request.get("capture_outputs"):
                storage, dtype = input_storage(actual_tensor.contiguous())
                name = f"outputs/{case_index}.npy"
                (output / "outputs").mkdir(exist_ok=True)
                np.save(output / name, storage, allow_pickle=False)
                record["outputs"].append({"file": name, "dtype": dtype, "storage_dtype": storage.dtype.name,
                                          "shape": list(storage.shape), "sha256": hashlib.sha256(storage.tobytes()).hexdigest()})
            expected, actual = host_array(expected_tensor), host_array(actual_tensor)
            if actual.shape != expected.shape:
                reject(record, case_index, f"Output shape {actual.shape} differs from {expected.shape}.")
                return record
            if actual_dtype != expected_dtype:
                reject(record, case_index, f"Output dtype {actual_dtype} differs from {expected_dtype}.")
                return record
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                reject(record, case_index, "This finite-input contract requires finite outputs.")
                return record
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0))
            record["max_absolute_error"] = max(error, record["max_absolute_error"])
            if validator:
                stage = "validation"
                outcome = validator.validate(tuple(torch_input(copy_input(value)) for value in case), actual_tensor)
                if not isinstance(outcome, dict) or not isinstance(outcome.get("passed"), bool):
                    raise TypeError("validate(inputs, actual) must return a dict with a boolean passed field.")
                json.dumps(outcome, allow_nan=False)
                record["validation"].append({"case": case_index, **outcome})
                passed = outcome["passed"]
            else:
                passed = bool(np.allclose(actual, expected, rtol=request["rtol"], atol=request["atol"], equal_nan=False))
                record["validation"].append({"case": case_index, "passed": passed, "metrics": {"max_absolute_error": error}})
            if not passed:
                record.update(status="incorrect", failed_case=case_index, failure_kind="validation", message="Output differs from the workload reference.")
                if validator:
                    record["message"] = str(outcome.get("reason", "Workload validator rejected the output."))
                return record
            if request.get("read_only_inputs") or request.get("poison_outputs"):
                stage = "diagnostics"
                diagnostics = runner.check(case, read_only_inputs=request.get("read_only_inputs", False),
                                           poison_outputs=request.get("poison_outputs", False))
                record["diagnostics"].append({"case": case_index, **diagnostics})
                if any(value["status"] == "failed" for value in diagnostics.values()):
                    record.update(status="incorrect", failed_case=case_index, failure_kind="validation",
                                  message="Buffer diagnostic failed.")
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
        from gpu2tensor.backends.ptx import AssemblyError
        failure_kind = "candidate_compile" if isinstance(error, AssemblyError) else "runtime_or_infrastructure"
        if isinstance(error, SyntaxError) and error.filename == str(directory / "candidate.py"):
            failure_kind = "candidate_compile"
        record.update(status="error", stage=stage, failure_kind=failure_kind, message=str(error))
        traceback.print_exc()
        return record


def main():
    directory = Path(sys.argv[1])
    record = evaluate(directory, sys.argv[2])
    (directory / "output/record.json").write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
