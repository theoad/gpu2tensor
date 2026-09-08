"""Explicit CPU plumbing check; not evidence of accelerator support."""

import platform
import time

import torch


class Runner:
    timing_method = "cpu_wall_clock"

    def __init__(self, module):
        self.module = module

    def identity(self):
        return {"device": platform.machine(), "purpose": "cpu_plumbing_check"}

    def run(self, case):
        return self.module.run(*(torch.from_numpy(value.copy()) for value in case))

    def benchmark(self, case, repetitions):
        samples = []
        for _ in range(repetitions):
            inputs = tuple(torch.from_numpy(value.copy()) for value in case)
            start = time.perf_counter_ns()
            self.module.run(*inputs)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        return samples

    def profile(self, case, output):
        raise RuntimeError("CPU is a plumbing check and supplies no accelerator profile.")


def prepare(module):
    return Runner(module)
