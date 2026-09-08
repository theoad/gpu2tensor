"""Triton execution, CUDA event timing, and a separate Proton profile."""

import json
import subprocess
import time

import torch
import triton

from gpu2tensor.profiles import proton_kernels
from gpu2tensor.tensors import copy_input, torch_input


class Runner:
    timing_method = "cuda_events_including_launch_gaps"

    def __init__(self, module):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this worker environment.")
        self.module = module
        self.metadata = {}

    def identity(self):
        return {**self.metadata, "measurement_policy": {"warmup_runs": 5, "inputs": "fresh_copy_before_each_call_outside_timing",
                "outputs": "allocated_inside_call", "cache": "uncontrolled", "working_set": "benchmark_case_0",
                "synchronization": "device_before_timing_end_event_after_call"}, "device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                "triton": triton.__version__, "compute_capability": list(torch.cuda.get_device_capability()),
                "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}

    def inputs(self, case):
        return tuple(torch_input(copy_input(value)).cuda() for value in case)

    def run(self, case):
        return self.module.run(*self.inputs(case))

    def check(self, case, *, read_only_inputs, poison_outputs):
        from gpu2tensor.diagnostics import fingerprint, report
        can_poison = hasattr(self.module, "output_poison")
        unchanged, results = True, []
        for poison in ([0x3f, 0xbf] if poison_outputs and can_poison else [None]):
            inputs = self.inputs(case)
            before = [fingerprint(value.cpu()) for value in inputs]
            if can_poison:
                self.module.output_poison = poison
            try:
                actual = self.module.run(*inputs)
                torch.cuda.synchronize()
            finally:
                if can_poison:
                    self.module.output_poison = None
            if read_only_inputs:
                unchanged &= before == [fingerprint(value.cpu()) for value in inputs]
            results.append(fingerprint(actual.detach().cpu().contiguous()))
        return report(read_only_inputs=read_only_inputs, poison_outputs=poison_outputs,
                      inputs_unchanged=unchanged, outputs_match=results[0] == results[-1] if can_poison else None)

    def benchmark(self, case, repetitions):
        for _ in range(5):
            self.module.run(*self.inputs(case))
        torch.cuda.synchronize()
        samples = []
        for _ in range(repetitions):
            # Restore inputs outside timing, including for in-place candidates.
            inputs = self.inputs(case)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.module.run(*inputs)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return samples

    def profile(self, case, output):
        import triton.profiler as proton

        inputs = self.inputs(case)
        self.module.run(*inputs)
        torch.cuda.synchronize()
        inputs = self.inputs(case)
        torch.cuda.synchronize()
        name = str(output / "proton")
        collection_started = time.perf_counter_ns()
        session = proton.start(name, context="shadow", backend="cupti")
        run_started = time.perf_counter_ns()
        try:
            with proton.scope("candidate"):
                self.module.run(*inputs)
            torch.cuda.synchronize()
            run_ms = (time.perf_counter_ns() - run_started) / 1e6
        finally:
            proton.finalize(session)
        collection_ms = (time.perf_counter_ns() - collection_started) / 1e6
        profile = json.loads((output / "proton.hatchet").read_text())
        kernels = proton_kernels(profile)
        (output / "events.json").write_text(json.dumps(kernels, indent=2))
        return {"provider": "proton_cupti", "collection": "separate_execution",
                "scope": "kernel_launch_aggregates", "completeness": "not_verified",
                "replay": "not_requested", "sampling": "not_reported", "dropped_events": None,
                "profiled_run_wall_ms": run_ms, "capture_wall_ms": collection_ms,
                "kernels": sum(int(item["metrics"].get("count", 1)) for item in kernels),
                "kernel_groups": len(kernels),
                "device_time_ns": sum(item["duration_ns"] for item in kernels),
                "artifacts": ["proton.hatchet", "events.json"]}


def prepare(module):
    return Runner(module)
