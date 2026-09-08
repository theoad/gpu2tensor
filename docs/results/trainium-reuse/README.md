# Trainium process reuse

2026-09-08, Trainium1 `trn1.2xlarge`, host `ip-172-31-64-161`, physical and logical
core 0. [Identity](environment.json): Torch 2.11.0+cu130, NKI
0.6.0+31049202112.g85070674, neuronx-cc 2.27.5334.0+f702b353. Runtime code was
unchanged from gpu2tensor `039dc4d`; a separate source directory and temporary
workers preserved the experiment's original `ae03dbc` endpoint.

## Request cost and profiles

[Four trials per mode](comparison.json) measured fused softmax on four independent
32x128 float32 input cases. Each request passed correctness, took 30 unprofiled
resident-runtime timings, and captured/decoded a separate native profile. A warmup
request preceded each group. Median whole-request time was 10.2943 seconds with
fresh processes and 2.8122 seconds with one reused child (3.66x ratio).

The fresh group ran before the reused group on the same core. This is a small,
grouped comparison susceptible to host drift, not an alternating/randomized
estimate or a kernel speedup. It includes network, serialization, imports,
compilation, validation and profiling. Neuron compilation still occurs per job;
no new NEFF compilation cache was added.

All eight measured profiles agreed on engine instruction counts and byte counts:
22 scalar, 25 vector, 21 tensor, 28 GPSIMD and 16 sync instructions; 16,384 HBM
bytes read and written. Timing/cycle counts can vary. Matching counts do not prove
trace completeness; dropped events remain unknown. The reused trials shared PID
14002 and matched the fresh group on inputs, core, SDKs and measurement policy.

## Ownership finding and controls

The initial alternating run failed when its next fresh process tried to initialize
NRT after the reused child had acquired core 0. An idle reused Neuron child retains
its assigned core. This stayed an infrastructure failure, without a timing label;
its original artifacts are preserved under `comparison/`. After restarting only
the temporary reused service, the fresh-first grouped comparison passed.

[Eight further controls](controls.json) verified:

- Four valid requests with exact output retention, input checks, output poison
  checks and native profiles passed. The child recycled after its configured
  eighth job (including preceding warmup/trials), changing PID 14002 to 14412.
- Invalid shared-HBM placement returned a structured candidate compilation error,
  no timing, and forced recycling. A valid request then passed in PID 14489.
- A candidate producing correct softmax output while modifying its input failed
  the input diagnostic and received no timing. The following valid profiled request
  passed in another child, PID 14591.

This qualifies short NKI reuse sequences and normal resource recycling on this
SDK. It does not qualify vendor Torch-Neuron reuse, other SDK versions, arbitrary
persistent global state, sustained memory behavior or device-fault recovery.
Fresh processes remain default. See [setup and ownership](../../process-reuse.md).

## Handback and evidence

Both temporary worker services used process-group cleanup and a ten-minute runtime
limit. They were explicitly stopped before returning the chip. Neuron reported
no remaining processes; the original worker stayed active on `ae03dbc`. The
experiment's 16:28:59 UTC termination guard was preserved.

Raw requests, native traces, exact outputs, source snapshots and setup/control
scripts are under ignored `artifacts/trainium-reuse/`, backed up to the operator's
`s3://gpu2tensor-dev-artifacts-arqozvlqbtub/trainium-reuse/results/` prefix. Local
checks cover the example CLI and rejection of alternating Trainium mode before
any connection. No runtime implementation or SDK installation changed for this
qualification.
