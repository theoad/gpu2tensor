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
