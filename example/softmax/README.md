# Check a complete kernel

```bash
python example/softmax/run.py --backend cpu --output artifacts/cpu-softmax
python example/softmax/run.py --backend cuda --endpoint http://127.0.0.1:8000 --profile --output artifacts/cuda-softmax
```

For a Trainium worker, use `--backend trainium` with its forwarded endpoint.
Four float32 input cases compare the submitted program with Torch softmax.
Successful accelerator runs save timing samples and optional native profiles.
The CPU mode checks evaluation plumbing and supplies no accelerator profile.

Read the [kernels and workload](../../python/gpu2tensor/examples/softmax.py) and
[example guide](../../docs/examples.md). Run `--help` for shape/output options.
