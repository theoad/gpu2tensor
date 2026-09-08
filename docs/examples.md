# Small learning examples

Start with softmax: four float32 input cases compare a submitted program with
`torch.softmax`. They cover normal inputs, larger magnitudes, zeros and a wide
ramp. Correctness checks all four; timing and profiling use the first case.
The included NKI program targets a single tile (the tested shape is 32 × 128),
not arbitrary tensor sizes. Failed candidates receive diagnostics and no timing.

Install the learner with `python -m pip install -e '.[learning,gym]'`.
For a first run without accelerator hardware, use the included recorded dataset:

```bash
python example/pretrain/train.py --dataset example/pretrain/measurements.json --device cpu --output artifacts/first-classifier
```

This reads 48 real A10G profile records and trains the same small classifier used
by live collection. It saves per-sample metadata, `report.json` and `classifier.pt`.
The dataset includes hardware/toolchain identity and input/source hashes; native
profile files are retained separately and are not part of the GitHub dataset.
The split, measured features and controls are identical to the original experiment.

Prepare workers and tunnels using [the operator guide](environments.md).

```bash
python -m gpu2tensor.examples.softmax --backend cuda --endpoint http://127.0.0.1:18000 --profile --output artifacts/cuda-softmax
python -m gpu2tensor.examples.softmax --backend trainium --endpoint http://127.0.0.1:18001 --profile --output artifacts/trainium-softmax
```

Both return the same result contract. CUDA currently captures kernel launch
aggregates with Proton/CUPTI. Trainium captures NEFF/NTFF artifacts and optionally
decodes byte counts, cycles and per-engine instruction counts with Neuron Explorer.
Native summaries remain available; counters from different vendors are not aliases.

## Observation-only collection

This path needs no Gym, rewards, actions or policy. Each completed evaluation is
yielded separately. Batching reads only as far ahead as the requested batch size.

```python
from gpu2tensor import Evaluator
from gpu2tensor.data import batches
from gpu2tensor.examples.softmax import candidate, workload

worker = Evaluator("http://127.0.0.1:18001")
results = worker.observe([candidate("trainium")], workload())
columns = ["latency_ms", "profile.measurements.hbm_read_bytes"]
for batch in batches(results, columns, batch_size=8):
    tensors = batch.tensors("mps")  # use "cpu" or "cuda" on other learners
    print(tensors["values"], tensors["valid"])
```

CPU tensors share the batch's NumPy storage. Moving to MPS/CUDA copies that batch.
Features are float32; exact large integer counters remain in `batch.records`.
Always consume the validity mask. A missing measurement is not a measured zero.
Save each result into an empty directory. This keeps artifacts from different
runs separate. Saving before batching keeps full source, inputs, diagnostics and native
artifacts for later reuse. `Result.load` reads metadata only.

The runnable observation-only learning example classifies two supplied GPU
programs from launch count and profiled device duration. It collects 48 examples
at four row counts, splits by input seed, and trains a tiny model on MPS or CPU.

```bash
python -m gpu2tensor.examples.pretrain --endpoint http://127.0.0.1:18000 --device mps --output artifacts/pretrain
python -m gpu2tensor.examples.pretrain --reuse --device mps --output artifacts/pretrain
```

The split holds out inputs, not program families. This is deliberately an easy
data-path test, not a compiler generalization result. It reports shuffled-label
and zero-measurement controls alongside the measured classifier.

## Source actions and feedback

```python
from gpu2tensor import Evaluator
from gpu2tensor.examples.softmax import candidate, workload
from gpu2tensor.gym import KernelEnv

env = KernelEnv(Evaluator("http://127.0.0.1:18001"), workload(), language="nki",
                reward=lambda result: float(result.correct),
                success=lambda result: result.correct)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(candidate("trainium").source)
print(info["result"].record)
```

The CUDA learning example uses this interface for a two-program bandit: one
program launches once per row; the other launches all rows together. A model
chooses complete source actions and learns from measured latency. It selects
between supplied programs; it does not generate a compiler or new source yet.

```bash
python -m gpu2tensor.examples.learn --endpoint http://127.0.0.1:18000 --device mps --steps 24 --output artifacts/learn
```

See [recorded results](results/README.md) for hardware, controls and limits.

## Upstream interfaces

The CUDA adapter uses [Triton Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton).
The Trainium adapter uses the installed NKI 0.6 compiler/runtime and
[Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/get-started.html).
Its compiler entry points are version-sensitive and checked against the exact
development image. Dependencies keep their own licenses; this package bundles
neither a vendor SDK nor a device driver.
