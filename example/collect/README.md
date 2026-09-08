# Bounded worker collection

```bash
python example/collect/run.py --backend trainium --endpoints http://127.0.0.1:18001 http://127.0.0.1:18002
```

Each endpoint must own a separate device or explicitly assigned device partition.
The example warms both workers and alternates serial/parallel trials over the same
correct affine program. It saves raw results and reports whole-request throughput.
See [the ownership and measurement guide](../../docs/next-slice.md).

To measure process startup savings, use `reuse.py` with two idle endpoints on the
same GPU, one fresh and one reused. This comparison submits jobs sequentially.
See [the reuse setup and command](../../docs/process-reuse.md).

`reuse.py --backend trainium` runs fresh jobs before reused jobs because an idle
Neuron child retains its core. Start the reused endpoint without any earlier jobs;
stop it completely before returning the core to another worker.
