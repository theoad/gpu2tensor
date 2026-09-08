"""Explicit CPU plumbing check; not evidence of accelerator support."""

import platform
import time

import torch

from gpu2tensor.tensors import copy_input, torch_input


class Runner:
    timing_method = "cpu_wall_clock"

    def __init__(self, module):
        self.module = module

    def identity(self):
        return {"device": platform.machine(), "purpose": "cpu_plumbing_check"}

    def run(self, case):
        return self.module.run(*(torch_input(copy_input(value)) for value in case))

    def check(self, case, *, read_only_inputs, poison_outputs):
        from gpu2tensor.diagnostics import fingerprint, report
        inputs = tuple(torch_input(copy_input(value)) for value in case)
        before = [fingerprint(value) for value in inputs]
        self.module.run(*inputs)
        return report(read_only_inputs=read_only_inputs, poison_outputs=poison_outputs,
                      inputs_unchanged=before == [fingerprint(value) for value in inputs])

    def benchmark(self, case, repetitions):
        samples = []
        for _ in range(repetitions):
            inputs = tuple(torch_input(copy_input(value)) for value in case)
            start = time.perf_counter_ns()
            self.module.run(*inputs)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return samples

    def profile(self, case, output):
        raise RuntimeError("CPU is a plumbing check and supplies no accelerator profile.")


def prepare(module):
    return Runner(module)
