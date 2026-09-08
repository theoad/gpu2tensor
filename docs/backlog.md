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

The measurement/worker slice is merged to `master`. Its acceptance checks:

- [x] Repeat CUDA profiles, including an input-mutating kernel; measure timing variance and collection overhead.
- [x] Add a second workload and evaluate held-out program families with repeated permutation controls.
- [x] Train a small policy through the Trainium Gym endpoint with measured rewards.
- [x] Measure collection throughput and add a bounded endpoint pool with ownership/backpressure tests.
- [x] Save remaining raw results; hand off the reserved worker after matmul qualification.

Evidence: [next-slice results](results/next/README.md) and
[CUDA qualification](results/cuda/README.md). AWS GPU Spot remained unavailable;
the user authorized a bounded Runpod fallback, which completed CUDA qualification. The shared Trainium worker
is now reserved for integration with the matmul experiment, within its shared
$100 campaign cap and 2026-09-09 06:00 UTC hard stop. Coordinate new launches.

Matmul client requests are tracked as GitHub issues #1 (exact BF16), #2 (PTX
launch/buffer contract), #3 (matched vendor Torch baselines), and #4 (workload-owned
validation or exact outputs). Keep their qualification evidence separate. Issue #5 adds opt-in buffer diagnostics.

BF16 transport, fixed validators and exact outputs pass local and Trainium checks;
read-only input and output poison checks pass on Trainium. Complete PTX launch
contracts, CPU assembly and CUDA device load/launch pass the initial qualification. The matched
Torch-Neuron baseline passes five shapes with the experiment validator and three
input families; the frozen experiment corpus remains the client's responsibility. See [contract](matmul-contract.md) and
[qualification evidence](results/matmul/README.md).

1. Extend CUDA validation from the initial three families to the experiment-owned
   frozen corpus. Initial PTX and strict vendor checks pass all five shapes; vendor
   defaults fail two shapes. Repeated profiles and input restoration pass on RTX 4090.
2. Qualify trace completeness and missing/dropped events before making completeness
   claims. CUDA currently returns launch aggregates.
3. Consider Neuron compilation reuse after qualifying its cache identity and
   ownership contract. NKI process reuse now passes [Trainium checks](results/trainium-reuse/README.md);
   vendor Torch-Neuron reuse remains unqualified. CUDA reuse and bounded PTX caching
   are implemented: warmed requests took 2.60 versus 0.54 seconds on the qualified
   RTX 4090 host. Fresh isolation remains default; neither the pool nor reuse makes
   this collection path accelerator-bound. See [reuse](process-reuse.md).
4. Qualify additional NKI/Torch-Neuron SDK versions. Compiler ABI extraction is
   version-sensitive; keep source, toolchain identity and native artifacts.
5. Extend input contracts to strides, multiple outputs and Torch graph extraction
   only with a concrete workload. Keep observation-only clients independent of Gym.
6. Add optional CUDA instruction/memory signals and stronger buffer diagnostics.
   Native code gets one CMake tree if an in-process component becomes necessary.

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

## NKI classification repair

Issue #7: the scoped `nki-0.6-buffer-placement-v1` policy now distinguishes two
confirmed frontend buffer-placement assertions from unknown runtime/compiler
faults. Exact negative replays and a valid control passed on the reserved
Trainium slot. Timing/execution/profile methods are unchanged. See
[policy and evidence](results/nki-diagnostics/README.md). Existing records can be
classified into separate hashed annotations without mutation or redispatch.

## NKI vector-transpose classification

Issue #9 adds a narrow saved-failure/compiler policy for the observed [64, 32]
vector-transpose rejection. Local checks and both controls pass on a newly qualified replacement after AWS
reclaimed the original Spot worker. The exact negative has no timing; the valid
128³ BF16 control passes six families with identical captured output files.
See [policy and qualification status](results/nki-transpose/README.md).

## Deferred CUDA timing condition

Issue #10 requests an optional, separately named controlled-launch replay/device
measurement for PTX and vendor Torch. Preserve the current event interval and
frozen campaign. Define matching allocation, restoration/residency and cache
policies; preinitialize events; qualify replay/capture, launch-delay controls and
both languages' outputs. Keep profiler durations separate and do not subtract
unmeasured overhead. Implement on an independently owned worker after the current
campaign, including explicit unsupported-capture/allocation limits.
