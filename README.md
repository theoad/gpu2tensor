# gpu2tensor

Accelerator measurements as tensors, and complete kernel source as actions.
Evaluate Torch workloads on NVIDIA/Triton or Trainium/NKI; learn from the results
on CPU, CUDA or Apple MPS. Licensed under AGPL-3.0-only.

## Install and try it

```bash
git clone https://github.com/theoad/gpu2tensor.git
cd gpu2tensor
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[learning,gym]'

# Train a tiny classifier on 48 recorded GPU profiles. No GPU or AWS needed.
python example/pretrain/train.py --dataset example/pretrain/measurements.json --device cpu --output artifacts/first-classifier

# Check the evaluation plumbing locally, without an accelerator profile.
python example/softmax/run.py --backend cpu --output artifacts/cpu-check
```

The classifier saves `report.json` and `classifier.pt`. Use a fresh output
directory for each collection run. On Apple silicon, pass `--device mps` to
train on MPS. This repository install is available now; PyPI publishing is pending.

## Observe an accelerator

On an operator-prepared CUDA host with gpu2tensor, Torch and Triton installed:

```bash
gpu2tensor-worker --backend cuda
```

The worker binds to `127.0.0.1:8000`. On the same host, use the endpoint directly;
for a remote host, forward port 8000 through your SSH/SSM tunnel. See the
[worker setup](docs/environments.md) for the qualified CUDA and Trainium images.
Workers execute trusted source. Operators provide drivers, SDKs and connections.

```python
from gpu2tensor import Evaluator
from gpu2tensor.examples.softmax import candidate, workload

worker = Evaluator("http://127.0.0.1:8000")
result = worker.evaluate(candidate("cuda"), workload(), profile=True)
print(result.correct, result.latency_ms)
result.save("artifacts/first-kernel")
```

For Trainium, start `gpu2tensor-worker --backend trainium` in the qualified NKI
0.6 environment and submit `candidate("trainium")`. To submit your own program,
use `Candidate.from_file("kernel.py", language="triton")` or `language="nki"`.

Observation-only collection needs no Gym, policy or reward:

```python
from gpu2tensor.data import batches

results = worker.observe([candidate("cuda")], workload())
for batch in batches(results, ["latency_ms", "profile.kernels"], batch_size=8):
    tensors = batch.tensors("cpu")  # also accepts "cuda" or "mps"
    print(tensors["values"], tensors["valid"])
```

CPU tensors share the batch's NumPy storage; accelerator transfer copies it.
Always consume the validity mask: unavailable counters are not measured zeros.
Timing uses an unprofiled run; profiling is a separate execution. CUDA currently
exposes launch aggregates. Trainium provides native instruction traces and selected
byte, cycle and engine counters. Trace completeness is not yet verified.

For multiple workers, `Pool(endpoints).observe(candidates, workload)` keeps one
job per endpoint and yields completed results with backpressure. Each endpoint
must own a separate device or assigned partition. See the
[pool example and measurements](docs/next-slice.md) for setup, ownership and limits.

## Submit actions through Gym

```python
from gpu2tensor.gym import KernelEnv

env = KernelEnv(worker, workload(), language="triton",
                reward=lambda result: float(result.correct),
                success=lambda result: result.correct)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(candidate("cuda").source)
print(info["result"].record)
```

The client defines correctness cases, rewards and episode success. Each action
submits a complete program. Incorrect programs receive diagnostics and no timing.

## Small learning examples

| Example | Run from the repository root | What it checks |
| --- | --- | --- |
| [Softmax](example/softmax/) | `python example/softmax/run.py --backend cuda --endpoint http://127.0.0.1:8000 --profile` | Torch reference correctness, timing and profiling; also supports Trainium |
| [Observation-only learning](example/pretrain/) | `python example/pretrain/train.py --endpoint http://127.0.0.1:8000` | Collect profiles and train a small classifier; supports offline replay |
| [Gym policy learning](example/gym/) | `python example/gym/train.py --endpoint http://127.0.0.1:8000 --steps 24` | Learn to choose between two complete kernels using measured feedback |
| [Affine workload](example/affine/) | `python example/affine/run.py --backend cpu` | An independent pointwise workload, also supported on CUDA/Trainium |
| [Worker collection](example/collect/) | `python example/collect/run.py --backend trainium --endpoints http://127.0.0.1:18001 http://127.0.0.1:18002` | Compare serial and bounded parallel observation collection |

The Gym example also accepts `--backend trainium`. The
[family-held-out experiment](docs/next-slice.md) adds repeated controls and feature
ablations; its results are more limited than the original input-only split.

These are MNIST-style starter exercises using accelerator measurements, not the
MNIST image dataset. They validate the learning data path; they do not demonstrate
compiler synthesis or generalization to unseen program families. See the
[results and controls](docs/results/README.md), including the weak shuffled-label
control in the initial classifier experiment.

Read the [example guide](docs/examples.md), [architecture](docs/architecture.md),
[development recipe](docs/development.md) and [work board](docs/backlog.md).

BF16 inputs, workload validators, exact output artifacts, and explicit PTX launch
contracts are described in [the matmul guide](docs/matmul-contract.md). PTX GPU
execution is awaiting hardware qualification.
