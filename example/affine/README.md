# Affine transform

```bash
python example/affine/run.py --backend cpu
python example/affine/run.py --backend trainium --endpoint http://127.0.0.1:18001 --profile --output artifacts/affine-trainium
```

Four input cases check `y = 0.5 * x + 0.25`. This pointwise workload is independent
of softmax's reductions. See [the implementation](../../python/gpu2tensor/examples/affine.py)
and [the next-slice guide](../../docs/next-slice.md).
