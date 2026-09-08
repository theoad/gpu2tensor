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

## Matmul baseline compiler environment

On the shared Trainium development host, `/opt/gpu2tensor/vendor-venv/bin/python`
provides Torch 2.9.0+cpu, Torch-Neuron 2.9.0.2.15.32035+de43f57c,
Torch-XLA 2.9.0 and neuronx-cc 2.27.5334.0+f702b353. The runtime worker stays in
its original NKI 0.6 / Torch 2.11 environment. Set
`GPU2TENSOR_TORCH_NEURON_PYTHON=/opt/gpu2tensor/vendor-venv/bin/python` on that
worker and install gpu2tensor editable in both environments. This is operator
setup; no installation occurs in a library call.

Pin `islpy==2026.1` in this compiler environment. Resolving the newer 2026.2.1
caused an internal `is_subset` compiler error on Torch matmul. Reverting to the
AMI's 2026.1 allowed compilation and hardware qualification. Other recorded
compiler dependencies: NumPy 2.5.3, protobuf 7.36.1, libneuronxla
2.2.17544.0+fb9962bf. The isolated environment leaves the runtime dependencies
unchanged. Compilation must use a writable per-job directory because the compiler
also writes a working-directory log.

Core 0's `gpu2tensor-worker.service` currently gets this setting through
`/etc/systemd/system/gpu2tensor-worker.service.d/vendor.conf`. Core 1 remains an
NKI worker. Reserve the whole chip when making timing comparisons; no concurrent
experiment submissions during qualification. The instance terminates on its
4-hour systemd guard, earlier than the shared campaign hard stop unless the
operator explicitly changes the lease. Coordinate any change with the campaign.

## Runpod CUDA fallback

The user authorized the existing $10 Runpod credit when AWS GPU capacity was
unavailable. Do not add credit or extend a lease implicitly. Keep pod identifiers,
SSH keys, API credentials, exact price, deadline and ownership in ignored `.local/`;
never upload operator credentials to the worker. The runtime has no Runpod dependency.

The qualified replacement used one RTX 4090 in US-NC-1 at a quoted $0.74/hour,
image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, a 20 GB ephemeral container
disk and no persistent volume. Request a minimum host CUDA version of 12.8 for
this image; an unconstrained placement reported 12.4 and was discarded before use.
Check the actual driver and a real Torch CUDA operation after connecting.

Use a local SSH alias such as `pod-gpu` with an ephemeral key and connection
multiplexing. Pod IPs and mapped SSH ports change. This image uses an externally
managed system Python, so create a venv with `--system-site-packages`, then install
the checkout editable with `--no-deps` to retain its vendor Torch/Triton versions.
The qualified versions are recorded in [the CUDA environment](results/cuda/environment.json).

The image's nginx already listens on 8001 and proxies 8000. Choose a dedicated
loopback port, such as 8011, and verify the worker's health JSON through the final
tunnel. Reuse qualification used 8010 separately. Never run both endpoints
concurrently on the same GPU; they were used only for sequential comparisons.

Before launch, install an operator cleanup guard with an absolute deadline and
verify its cloud API access. This session's local launchd guard checks every minute
and deletes only the named pod in its lease. It depends on the operator Mac being
awake and online. Keep it active through an explicit experiment handoff; transfer
the lease, accounting and guard paths together. Back up artifacts before expiry,
and explicitly delete idle pods when no owner needs them. Deleting an ephemeral
pod discards its local files. Keep failed placement and deleted-pod costs in the
same accounting ledger; a replacement does not reset the budget or deadline.
