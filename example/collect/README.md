# Bounded worker collection

```bash
python example/collect/run.py --backend trainium --endpoints http://127.0.0.1:18001 http://127.0.0.1:18002
```

Each endpoint must own a separate device or explicitly assigned device partition.
The example warms both workers and alternates serial/parallel trials over the same
correct affine program. It saves raw results and reports whole-request throughput.
See [the ownership and measurement guide](../../docs/next-slice.md).
