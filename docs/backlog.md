# First slice

Accepted: NVIDIA CUDA and Trainium; complete kernel source/IR actions; simple
synchronous clients; observation-only collection; optional Gym; local MPS learner.

## Acceptance

- [x] Provision one GPU and one Trainium Spot worker with pinned images.
- [x] Compile and validate supplied softmax kernels on both real devices.
- [x] Return uninstrumented timings and separate native profiling artifacts.
- [x] Collect bounded tensor batches without Gym.
- [x] Exercise Gym candidate-feedback and a small real-measurement learning run.
- [x] Record toolchain versions, measurements, tests, and limitations.

## Evidence

See [results](results/README.md), [examples](examples.md) and
[operator setup](environments.md). Both real accelerators passed four softmax
cases. GPU observation-only learning and a 24-action MPS bandit ran successfully.
Trainium Gym rejected a wrong source and accepted the correct source. Ten local
tests and an isolated wheel install passed. Raw traces and reports are retained.

## Next small slices

1. Rerun the final CUDA warmup-input restoration on hardware, then measure timing
   variance and profile overhead. Qualify missing/dropped events before promising
   complete traces. Current CUDA observations are launch aggregates.
2. Add an independent workload family and stronger learning controls; demonstrate
   Trainium policy training. Keep compiler generalization claims out of smoke tests.
3. Qualify NKI compiler integration beyond SDK 0.6. Its current frontend/compiler
   entry points are version-sensitive. Preserve raw vendor artifacts through changes.
4. Add a bounded endpoint pool with explicit device ownership when one-worker
   throughput is measured to be the bottleneck. No opaque async trajectory IDs.
5. Extend input contracts to strides, multiple outputs and Torch graph extraction.
   Add PTX/other IR submission only with a real example and correctness contract.
6. Add GPU instruction/memory signals as a distinct, optional collection mode.
   Native code gets one CMake tree if this requires an in-process component.

TPU and AMD are deferred. There is no common accelerator ISA, cloud scheduler,
replay buffer or compiler synthesis model in this first slice. PyPI publishing
remains separate release work.

## GitHub source release

The public repository is `theoad/gpu2tensor`. The README includes checkout/install,
an offline classifier quickstart, live observation/tensorization and Gym snippets.
`example/` contains runnable launchers for softmax, profile classification and Gym,
plus 48 recorded GPU samples for a hardware-free first run. Implementations retain
one import identity under `python/gpu2tensor/examples`.

Publication checks: the offline classifier completed on CPU, the CPU softmax
quickstart passed all four cases, the Gym launcher exposes its CLI, ten tests pass,
and documentation links resolve. Ignored local state, credentials, raw artifacts
and virtual environments are excluded from the source release. No AWS workers
were launched for publication.
