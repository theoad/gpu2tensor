# Learn from profiles

Train a two-class linear model from kernel launch count and profiled duration:

```bash
python example/pretrain/train.py --dataset example/pretrain/measurements.json --device cpu --output artifacts/classifier
```

The included dataset contains metadata from 48 real A10G evaluations: two supplied
softmax programs, four shapes and six input seeds. It includes hardware/toolchain
identity and source/input hashes; raw vendor traces are not bundled. Data is
provided under this repository's AGPL-3.0-only license. The original acquisition
command and limitations are in [the guide](../../docs/examples.md).

For live collection, replace `--dataset ...` with `--endpoint http://127.0.0.1:8000`.
For another training run from your saved results, use `--reuse --output artifacts/classifier`.
Choose CPU, CUDA or MPS with `--device`. Outputs are `report.json`, `classifier.pt`
and per-sample records. The report includes held-out-input accuracy, shuffled-label
and zero-measurement controls.

Read the [full training loop](../../python/gpu2tensor/examples/pretrain.py) and
[recorded results](../../docs/results/README.md). This small dataset holds out
inputs, not program families; its initial shuffled-label control was weak.
