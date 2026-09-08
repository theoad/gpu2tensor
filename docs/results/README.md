# First slice evidence

These are development measurements from September 2026, not performance claims
for other hosts. Hardware identity, input/source hashes, raw latency samples and
profiling summaries are in [hardware.json](hardware.json). Exact source, raw
profiles, checkpoints and inputs where retained are in ignored `artifacts/`,
backed up to the development S3 bucket. The earliest GPU runs predate inclusion
of `request.zip`; their deterministic workload seeds and shapes are recoverable
from the example and directory names. Later results retain the complete request.

## Hardware correctness and observation

| Check | Host | Result |
| --- | --- | --- |
| CUDA softmax, fused and one launch per row | ip-172-31-65-170, A10G, g5.xlarge | Four float32 cases pass; separate Proton profiles report 1 and 32 launches |
| CUDA incorrect output control | same A10G host | Rejected; no latency label |
| Trainium softmax | ip-172-31-64-151, Trainium1, trn1.2xlarge | Four cases pass; maximum absolute error 1.91e-6; NEFF/NTFF retained |
| Trainium decoded trace | same Trainium host | 16,384 HBM bytes read and 16,384 written; core cycles, SBUF bytes and engine instruction counts present |
| Trainium Gym | same Trainium host | Zero-output kernel rejected; correct source accepted and terminates through the client callback |
| Heterogeneous feature batch | local macOS 15.7.3 arm64, MPS learner | 2 × 3 batch transfers to MPS; CUDA-only and Trainium-only counters have opposite validity masks |

The CUDA bandit baseline medians were 0.855040 ms (one launch per row) and
0.051200 ms (fused), measured with CUDA events including host launch gaps on the
A10G host above. These small Python-launched kernels are sensitive to host gaps.
Trainium timing uses a different method: host round trips with resident device
tensors. Do not use these numbers as a cross-device speed comparison. Raw samples
and explicit methods are preserved for both devices.

Profiling uses separate executions. Neither backend has verified trace
completeness or dropped-event counts. CUDA v0 measurements are launch aggregates,
not instruction or memory traces. Trainium exports native instruction traces;
only a small set of counters with explicit units is tensorized. Instrumentation
overhead has not been characterized.

## Learning checks

[The bandit report](mps-bandit.json) records 24 real CUDA evaluations driven by a
policy trained on MPS. All actions passed correctness. Probability assigned to
the supplied fused program rose from 0.5 to 0.987676. This validates source
actions, feedback, rewards and policy updates; it does not synthesize new programs.

[The observation-only report](mps-pretrain.json) contains 48 measured profiles,
32 training samples and 16 test samples. A tiny MPS classifier reached 100% on
held-out input seeds; removing measurements yielded 50%. However, the single
shuffled-training-label control still reached 87.5%. The small, easy task has
weak controls and holds out no program families. Treat it only as a learning
data-path smoke check. A compiler-learning benchmark needs independent program
families, more seeds and stronger controls before making generalization claims.

[The Trainium Gym report](trainium-gym.json) records both outcomes and the mixed
vendor batch validity mask. Trainium hardware evaluation and source actions are
validated; a Trainium-trained policy has not yet been demonstrated.

## Local verification

Ten tests pass: correct/incorrect results, oracle isolation from input mutation,
syntax errors, process timeout, bounded feature consumption, CPU storage sharing,
Gym lifecycle, Proton scope decoding, strided-input rejection and observation-only
imports without Gym. A clean wheel excludes tests/operator tooling and was
installed outside the checkout into a fresh NumPy-only environment. Client
imports and request construction work there without Torch or vendor SDKs.

The GPU shutdown timer fired after the completed hardware runs. Subsequent CUDA
changes extracted its tested decoder and restored inputs after profile warmup;
those changes have local checks but have not been rerun on the A10G. The final
Trainium profiling path was exercised through the remote Gym endpoint.
