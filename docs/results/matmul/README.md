# Matmul contract qualification

2026-09-08, Trainium trn1.2xlarge `ip-172-31-64-161`, physical core 0,
AMI ami-0f8b952eba2c08d62; NKI 0.6.0+31049202112.g85070674,
neuronx-cc 2.27.5334.0+f702b353; Neuron runtime 2.34.10.

A 32x32 BF16 matmul using FP32 PSUM returned all 1024 elements exactly equal to
an FP64 reference rounded to BF16, across the Mac-to-worker boundary. Inputs used
multiples of 1/8, so this checks transport and execution without a tolerance
ambiguity. Native NTFF and summary capture passed. A deliberately rejecting
validator retained the output bits and skipped timing. The repeated run with
read-only inputs and two output poison patterns passed both buffer checks.
This does not establish bounds, lifetime, race safety, or all-shape correctness.
Raw runs are in `artifacts/next/bf16*` in the development checkout.

The experiment's scalar BF16 PTX seed assembled on the same host's CPU with
ptxas CUDA 12.8.93 for sm_80. A deliberately invalid instruction produced a
recognized source diagnostic. Source, cubin, hashes and logs are retained.
There was no CUDA device execution: GPU Spot capacity remains unavailable.

Local tests cover all 65,536 BF16 storage patterns, exact output retention,
validator gates and replacement tolerance, legacy float wire compatibility,
input mutation, unsupported diagnostics and explicit PTX launch validation.


## Vendor baseline and strict validator

The official Torch-Neuron BF16 matmul compiled and ran on the same Trainium host
and core above. Its compiler environment uses Torch 2.9.0+cpu, Torch-Neuron
2.9.0.2.15.32035+de43f57c and the same neuronx-cc version as the NKI worker.
It executes through the same resident NRT runner, with matching input restoration,
output reuse, synchronization, working set and uncontrolled cache policy.
[The 32x32 policy comparison](matched-policy.json) checks these fields and exact
input hashes against a fresh NKI matmul run. Their single-trial medians are too
close to claim a speedup; host/runtime overhead dominates these small calls.

[Five shape checks](vendor-shapes.json) passed the experiment's unchanged
FP32-accumulation-interval/BF16-rounding validator: 32x32x32, 128x128x128,
256x256x256, 128x256x128, and 127x193x65. Each uses three locally generated input
families (normal, zero, cancellation), seed 203, with read-only and output poison
checks. This is initial instrumentation/dependency qualification, not the
experiment's frozen six-family search or held-out corpus. The experiment must
run that corpus and independently recheck saved outputs after worker handoff.

[The profiled 32x32 vendor result](vendor-accepted.json) also passed exact BF16
comparison on multiples-of-1/8 inputs. A deliberate validator rejection skipped
timing. [A mutating NKI control](mutation-rejected.json) returned correct values,
then overwrote A; the input diagnostic caught it and blocked timing.

The fresh compiler environment initially failed because it installed islpy
2026.2.1. Pinning the qualified AMI's 2026.1 fixed the internal compiler error.
The failed request remains an infrastructure/unknown result, not a source label.
Compilation also now uses a writable per-job directory. Raw request, output,
NEFF, HLO/ABI metadata, dependencies and compiler logs are retained. The operator
S3 backup prefix is `s3://gpu2tensor-dev-artifacts-arqozvlqbtub/next/`.

CUDA vendor defaults and the separate `precision="strict"` variant are implemented,
but neither has new hardware qualification in this slice. Both require the client
validator; a failing default baseline must not be silently replaced by strict mode.

Final local validation: 39 tests pass. The wheel builds and its client imports
are checked in an isolated environment without Torch or vendor SDKs.
