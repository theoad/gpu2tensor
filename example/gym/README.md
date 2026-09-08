# Learn with source actions

With a CUDA worker forwarded to local port 8000:

```bash
python example/gym/train.py --endpoint http://127.0.0.1:8000 --steps 24 --output artifacts/policy
python example/gym/train.py --backend trainium --endpoint http://127.0.0.1:18001 --steps 24 --output artifacts/trainium-policy
```

The policy chooses between two supplied softmax programs. Each action submits
complete source, checks it against Torch, and receives a reward from measured
latency. The default learner uses MPS when available, otherwise CPU. Pass
`--device cuda` or `--device cpu` explicitly as needed.

Outputs include each evaluation, `report.json` and `policy.pt`. In the initial
A10G/MPS run, probability assigned to the fused program rose from 0.5 to 0.988.
This is a two-program bandit, not source synthesis. See the
[training loop](../../python/gpu2tensor/examples/learn.py),
[guide](../../docs/examples.md) and [results](../../docs/results/README.md).
