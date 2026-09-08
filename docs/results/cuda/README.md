# CUDA qualification and startup cost

2026-09-08, Runpod RTX 4090, container host `5489e0244f7e` in US-NC-1.
[Environment identity](environment.json): Torch 2.8.0+cu128, Triton 3.4.0,
NVIDIA driver 570.195.03, ptxas 12.8.93. All timings below use this host alone;
no cross-device or Trainium speedup comparison is made.

## Profiles and input restoration

[Three paired trials per program](fresh-profiles.json) passed correctness and
profiling for rowwise softmax, fused softmax and an input-mutating guard. The
non-mutating programs produced the expected 32 and 1 launch aggregates. The guard
rejects an input carrying its prior sentinel, checking restoration across
correctness, warmup, timing and profiling. Each trial uses 30 timing repetitions.

Fresh-process unprofiled timing medians were 0.7173 ms (rowwise), 0.04491 ms (fused)
and 0.08397 ms (guard), with sample coefficients of variation 5.4%, 8.9% and 4.5%.
CUDA events include host launch gaps. Whole-request profiling deltas were noisy:
the fused pairs added 0.062–0.219 seconds, while other programs had mixed signs.
Negative differences do not mean profiling is free or speeds execution up.
Native profile capture time and individual timings are retained in the reports.
Trace completeness and dropped-event counts remain unverified.

## BF16 matmul

[Five shapes](matmul-shapes.json) were checked against the experiment's unchanged
FP32-accumulation-interval/BF16-rounding validator: 32x32x32, 128x128x128,
256x256x256, 128x256x128 and 127x193x65. Each uses normal, zero and cancellation
inputs with seed 203, exact BF16 transport, captured output bits, read-only input
checks and output poison checks. This is three-family integration qualification;
it is not the experiment's frozen six-family corpus or a search result.

- Scalar PTX passed all five shapes, including module load, launch and profiling
  on the 32x32 case.
- Explicitly strict vendor Torch passed all five shapes and the 32x32 profile.
- Vendor defaults failed the normal family at 32x32x32 and 127x193x65 under the
  client validator. Those results retain outputs and receive no timing. The other
  three shapes passed. Strict mode remains a separately named/source-hashed action.

Early records contain the stale metadata string `cuda_runtime_qualification:
pending`; their actual checks and profiles succeeded. Records remain immutable.
The source now identifies `execution_api: cuda_driver` instead of a hard-coded
qualification assertion. Qualification belongs in evidence, not a backend flag.

## Process reuse and assembly cache

[Four alternating, warmed request pairs](process-comparison.json), with the same
inputs, device and measurement policy, reduced median fused-softmax request time
from 2.6043 to 0.5442 seconds (4.79x). Both modes requested profiling. The reused
trials shared one PID and returned one expected kernel each. These include network,
serialization, imports, checks and profiling; they are not kernel speedups or a
claim that collection is accelerator-bound. This is a small, single-host sample.

[Two additional paired trials per program](reused-profiles.json) passed profiling
and input restoration with the reused child. [PTX controls](ptx-cache.json) passed
cold assembly, a cache hit with identical cubin hash, and a valid request following
an invalid-opcode rejection. The rejection was `candidate_compile`, received no
timing and forced a new child PID. In this one assembly pair, assembly/materialization
took 15.97 ms on a miss and 1.05 ms on a hit; version probing is outside that interval.
This does not measure cold SDK startup or general compiler-cache speedup.

Fresh processes remain the default. Reuse is opt-in, has weaker state isolation,
and is not yet hardware-qualified on Trainium. See [the operating contract](../../process-reuse.md).

## Artifacts and failed setup

Raw requests, exact outputs, logs, native profiles and a source snapshot are under
ignored `artifacts/runpod/`, backed up to the operator's
`s3://gpu2tensor-dev-artifacts-arqozvlqbtub/runpod/` prefix. Public JSON reports
retain the source/input hashes needed to connect these checks to their results.

An initial reuse comparison hit the image's nginx listener on port 8001, which
proxied the fresh worker. That run is preserved under `reuse/` and excluded from
the reported comparison. The corrected example asserts actual process mode and
uses dedicated worker ports 8011 and 8010 (`reuse-verified/`). An initial pod with
an older host driver never became usable and was deleted; only the replacement
with a CUDA 12.8 minimum was used for these results.
