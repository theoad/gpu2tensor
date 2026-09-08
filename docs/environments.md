# Development runners

The first-slice CUDA and Trainium workers are both confirmed terminated. The
development stack and S3 bucket remain, holding source snapshots, a wheel and
results. Local reports are under `artifacts/` and `docs/results/`. Use the launch
recipe below for another hardware session.

Use `tools/gaws` for short operator commands after creating `.venv` and installing
the project editable. Add `tools/` to your shell PATH to use `gaws` directly.
AWS tooling is excluded from distribution wheels. The library does not provision
or install dependencies.

Region: us-east-1. The local AWS CLI default profile is already authenticated.
GPU Spot quota: 8 vCPUs. Trainium Spot quota: 8 vCPUs. On-Demand quota: zero for
both. Start with g5.xlarge and trn1.2xlarge. Freeze the actual device/toolchain in
each result. Region offerings do not guarantee Spot capacity.

The development stack has an S3 artifact bucket, SSM instance profile, and a
security group with no inbound ports. Source and results move through S3/SSM.
Instances use encrypted ephemeral root volumes and a four-hour shutdown timer.
Spot instances terminate on shutdown/interruption; completed artifacts must be
uploaded before cleanup. The operator should terminate idle runners explicitly.
The local inventory is `.local/aws.json`; it is never part of a public commit.

The learner may use MPS on the Mac. Measure targets without concurrent training.
Restart an interrupted candidate from its inputs. Never label a partial profile
as complete, and keep records from different hardware/toolchains separate.

## Operator commands

From the repository root, with AWS CLI and its Session Manager plugin available:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[learning,gym,test]'
tools/gaws init
tools/gaws status
tools/gaws launch cuda --zone us-east-1f
tools/gaws launch trainium --zone auto
tools/gaws status
tools/gaws sync cuda
tools/gaws start cuda
tools/gaws result --role cuda
tools/gaws sync trainium
tools/gaws start trainium
tools/gaws result --role trainium
```

Wait for CloudFormation `CREATE_COMPLETE` before launch, and SSM `Online` before
sync. `start` returns an SSM command ID; inspect it with `result` until it succeeds.
It installs this checkout into the image's existing environment and starts a
loopback worker as `ubuntu`. It does not install drivers or vendor SDKs. When
using both workers concurrently, pass the explicit command ID to `result`.
`sync` waits for completion, or reports the command to inspect before proceeding.

In separate terminals, keep these tunnels open while the learner runs:

```bash
tools/gaws tunnel cuda --port 18000
tools/gaws tunnel trainium --port 18001
```

Inspect hardware with `tools/gaws exec cuda 'nvidia-smi'` or
`tools/gaws exec trainium '/opt/aws/neuron/bin/neuron-ls'`.

After saving results locally or in the artifact bucket:

```bash
tools/gaws terminate cuda
tools/gaws terminate trainium
tools/gaws status
```

The stack and retained S3 bucket survive worker termination. Reuse the inventory
for later launches. S3 storage continues to incur charges until removed. Inventory
pins the selected AMIs; regenerating it resolves current images and requires
qualifying their SDKs again. Region and account quotas are development observations,
not portable defaults for clients.

## Qualified environments

| Role | Image | Interpreter | Key versions |
| --- | --- | --- | --- |
| A10G, g5.xlarge | ami-09a2a137d5bacdb1c | /opt/pytorch/bin/python | Python 3.13.15, Torch 2.10.0+cu130, Triton 3.6.0, NVIDIA driver 595.91.07 |
| Trainium1, trn1.2xlarge | ami-0f8b952eba2c08d62 | /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/python | Python 3.12.3, Torch 2.11.0+cu130, NKI 0.6.0+31049202112.g85070674, neuronx-cc 2.27.5334.0+f702b353 |
| Mac MPS learner | local .venv | .venv/bin/python | Python 3.14.6, Torch 2.14.0, NumPy 2.5.3, Gymnasium 1.3.0 |

Trainium uses `NEURON_RT_VISIBLE_CORES=0` and
`NEURON_PLATFORM_TARGET_OVERRIDE=trn1`. Add the SDK environment's `bin` and
`/opt/aws/neuron/bin` to PATH; `start` does this. Neuron Explorer
2.32.0.498-b1f3998 and runtime 2.34.10 (ac18d) decoded the tested trace. No
simulator was used for reported accelerator results.
