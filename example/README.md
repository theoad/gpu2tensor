# Small learning examples

Run these from the repository root after the README installation steps:

| Example | Entry point | Hardware needed |
| --- | --- | --- |
| [Softmax correctness](softmax/) | `python example/softmax/run.py --backend cpu` | CPU, or a CUDA/Trainium worker for accelerator checks |
| [Profile classification](pretrain/) | `python example/pretrain/train.py --dataset example/pretrain/measurements.json --device cpu` | None beyond the learner for the recorded dataset; CUDA worker for new profiles |
| [Gym policy learning](gym/) | `python example/gym/train.py --endpoint http://127.0.0.1:8000` | CUDA worker; CPU, CUDA or MPS learner |

Each entry point accepts `--help`. The launchers import one implementation from
`python/gpu2tensor/examples`; there are no duplicate training loops or import-path
changes. See [the guide](../docs/examples.md) for methods and expected outputs.
