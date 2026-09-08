# Measurement and collection results

All new hardware results below use Trainium1 on `ip-172-31-64-161`, a
`trn1.2xlarge` in us-east-1f with AMI `ami-0f8b952eba2c08d62`, NKI 0.6 and Neuron
compiler 2.27.5334.0+f702b353. The CUDA script is implemented but not qualified:
small G4dn/G5/G6 Spot launches failed, On-Demand GPU quota is zero, and the four
other checked regions have zero GPU Spot quota. No GPU instance was launched.
G4dn is not suitable for the newly accepted BF16 matmul campaign.

## Learning

[The Trainium policy](trainium-policy.json) completed 24 correct source actions
with an MPS learner. Probability of selecting the batched softmax rose from 0.5
to 0.986966. Initial same-host medians were 0.179192 ms for the rowwise program
and 0.071740 ms for the batched program, using host round trips with resident
tensors. This is a supplied-program bandit, not instruction synthesis.

[The family experiment](trainium-families.json) collected 48 real profiles across
softmax and an independent affine workload. All passed their four correctness
cases. Training on affine and testing softmax reached 100%; the reverse direction
reached 75%. Each fold uses different workload families and input seeds, 24 training
and 24 test samples, two measured instruction counters and 20 permutation controls.
The permutation statistics for held-out cross-entropy were 1/21 and 6/21 respectively;
the reverse direction does not demonstrate a clear improvement over the controls.

The first run exposed an actual normalization bug: a constant training counter
was divided by a tiny fallback scale, amplifying unseen values and untrained
weights. The fixed preprocessing masks constant training counters, without looking
at held-out labels or fitting held-out statistics. The original report is retained
in the raw artifact directory. A regression test covers this case.

## Collection

[The collection report](trainium-collection.json) compares four profiled affine
jobs, after warmup, with one worker versus two separately assigned physical cores
on the same chip. Serial trials took 39.10 and 37.06 seconds; pool trials took
21.18 and 32.18 seconds (1.85× and 1.15×). All jobs passed correctness and profiling.
The cores share host CPU and chip resources, and two trials are a small sample.

Preparation/checking takes roughly 6–14 seconds per candidate while native call
medians are around 0.08 ms. The pool improves collection throughput but does not
make this path accelerator-bound. SDK/process startup and compilation remain
important targets for further work.

Core qualification caught and corrected a second integration bug: after assigning
physical core 1 through `NEURON_RT_VISIBLE_CORES=1`, nrtpy must address logical core
0. Both workers now report their physical and logical indices. Initial failed
artifacts are preserved. Ownership, bounded consumption, completion order, iterator
close/error handling and normalization are covered by 14 passing local tests.

Raw artifacts are under ignored `artifacts/next/`. Each new result retains the
backend source/hash, exact request, diagnostics and native profiles. Native trace
completeness and dropped-event counts remain unverified.
