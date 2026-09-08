# Measurement and worker collection

This slice adds an independent affine workload, repeated controls, Trainium policy
training and bounded observation collection. Hardware results are recorded in the
work board as each check completes. A prepared script is not a hardware result.

## CUDA qualification

```bash
python example/softmax/characterize.py --endpoint http://127.0.0.1:18000 --trials 5
```

The script checks fused, rowwise and input-mutating programs. The mutation guard
fails if warmup, timing or profiling reuses a modified input buffer. Matched
profiled/unprofiled requests alternate order after a warmup request. Reports retain
all latency samples, variation, paired whole-request deltas, profiled run wall time
and capture wall time. Whole-request costs include SDK import, compile/cache,
correctness, profiling, transfer and serialization. Capture wall time includes
profiler start/finalize; its run wall time covers the synchronized invocation.
These are distinct from CUDA event timing. They are not pure instruction costs.

Known launch counts check the small examples' profiles. Completeness and dropped
events remain unverified; matching these counts cannot prove lossless tracing.

## Learning across program families

```bash
python example/pretrain/families.py --backend trainium --endpoint http://127.0.0.1:18001 --permutations 20 --output artifacts/families
python example/pretrain/families.py --backend trainium --reuse --permutations 20 --output artifacts/families
python example/gym/train.py --backend trainium --endpoint http://127.0.0.1:18001 --steps 24 --device mps --output artifacts/trainium-policy
```

The family experiment collects 48 profiles: softmax and affine, two complete
programs per family, four row counts and three seeds per family. Both folds hold
out the whole other workload family and its input seeds. Normalization fits only
training data. Labels describe rowwise versus batched execution, not semantics.
Only measured counters enter the classifier; names, source hashes and input
identifiers do not. CUDA uses launch count/duration; Trainium uses scalar/vector
instruction counts. Records retain exact inputs, programs and backend source.

Each fold reports cross-entropy and accuracy, 20 shuffled-training-label controls,
zero features, and individual feature ablations. The reported permutation statistic
uses held-out cross-entropy with the standard plus-one correction. High accuracy
with near-uniform probabilities can occur in these tiny controls. Two hand-written
families still cannot establish compiler generalization.

The Trainium Gym example submits complete rowwise or batched NKI softmax source.
It uses the same client-owned reward and episode interface as CUDA, with a CPU,
CUDA or MPS learner. This is selection between supplied programs, not synthesis.

## Pool ownership and backpressure

```python
from gpu2tensor import Pool
from gpu2tensor.data import batches
from gpu2tensor.examples.affine import candidate, workload

pool = Pool(["http://127.0.0.1:18001", "http://127.0.0.1:18002"])
results = pool.observe([candidate("trainium") for _ in range(8)], workload())
for batch in batches(results, ["latency_ms"], batch_size=4):
    print(batch.tensors("cpu"))
```

Each endpoint owns one separate device or assigned partition. All endpoints in
one call must support the submitted source language. One evaluation per endpoint
is active or waiting for consumption; the pool never queues the whole input
iterator. Results arrive as workers finish and retain `worker_endpoint`, candidate
name and source/input hashes. Input order is not preserved. Tensor batching keeps
its existing independent bound and missing-value mask.

Consume the iterator fully, or call `results.close()` in a `finally` block when
stopping early. Closing waits for already-running calls (bounded by their worker
timeouts); it does not cancel remote execution. Transport errors propagate after
in-flight work finishes. One pool permits one active observation iterator. The
operator must prevent separate pools, endpoint aliases or Gym clients from sharing
the same physical device concurrently. Gym remains attached to a chosen Evaluator;
this pool does not schedule interactive episodes or store recurrent states.

To assign the two cores on one development Trainium1 chip:

```bash
tools/gaws start trainium --core 0 --port 8000
tools/gaws result --role trainium
tools/gaws start trainium --core 1 --port 8001
tools/gaws result --role trainium
tools/gaws tunnel trainium --port 18001 --remote-port 8000
# In another terminal:
tools/gaws tunnel trainium --port 18002 --remote-port 8001
```

The operator assigns one physical core with `NEURON_RT_VISIBLE_CORES`. Neuron maps
that visible core to logical core zero inside the process. Each result reports
both indices. The cores share host CPU and chip
memory resources: parallel throughput must be measured, not assumed. Compare with
`python example/collect/run.py --backend trainium --endpoints http://127.0.0.1:18001 http://127.0.0.1:18002`.
Its serial/parallel trials run after warmup and retain per-job preparation, profile
and kernel durations. Do not mistake faster collection for a GPU-bound pipeline.
