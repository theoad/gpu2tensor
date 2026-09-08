# Matmul integration

The experiment owns its reference, validator and rewards. Workload cases accept
contiguous NumPy arrays or contiguous CPU Torch tensors. BF16 is transported as
uint16 storage with a separate logical dtype; values are not rounded to float16.
The worker needs matching protocol 2 support. Protocol 1 float clients still work.

```python
workload = Workload(reference_source, [(a.bfloat16(), b.bfloat16())],
                    validator=validator_source, capture_outputs=True,
                    read_only_inputs=True, poison_outputs=True)
```

A validator module defines `validate(inputs, actual)` and returns
`{"passed": bool, "reason": str, "metrics": dict}`. Inputs and actual are CPU
Torch tensors preserving logical dtype. Shape, dtype and finite-output checks run
first. The validator replaces the default allclose rule, then timing runs only on
accepted outputs. This is trusted code in the evaluation process, not an isolated
oracle. Keep independent finalist validation in the client.

`record.validation` has zero-based case indices. Rejections have status
`incorrect`, `failed_case`, and `failure_kind="validation"`. Source syntax or
recognized assembler diagnostics can have `failure_kind="candidate_compile"`.
Runtime, transport, timeout and ambiguous compiler failures do not establish that
source is invalid: clients should leave their semantic outcome unknown.

With `capture_outputs`, `record.outputs` describes each saved output's file,
shape, logical dtype, storage dtype and SHA256 of contiguous storage bytes.
Artifacts include rejected outputs, request.zip, sources, validator, and engine
source hashes. BF16 `.npy` artifacts contain uint16 bits. Rehydrate with
`torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)`. `Result.save()` keeps
these files. `Result.load()` reads metadata only.

Optional input checks compare buffer contents before/after another execution.
Neuron and PTX output checks fill owned buffers with two distinct byte patterns
and compare results from separate runs. These empirical checks are outside timing;
they do not prove all bytes are written or that memory use is safe. Opaque Triton
and Torch return allocations cannot be prefilled, so output poisoning is reported
as unsupported. Bounds, lifetime and race detection remain unsupported. Inspect
`record.diagnostics`; a requested diagnostic is not automatically a covered one.

## Complete PTX actions

```python
from gpu2tensor import Candidate
from gpu2tensor.ptx import Launch, Input, Output, Scalar

candidate = Candidate(ptx_source, "ptx", name="matmul", launch=Launch(
    entry="matmul", grid=((m*n + 127)//128, 1, 1), block=(128, 1, 1),
    arguments=(Input(0, alignment=2), Input(1, alignment=2),
               Output((m, n), "bfloat16", alignment=2),
               Scalar("uint32", m), Scalar("uint32", n), Scalar("uint32", k)),
    shared_memory_bytes=0))
```

Arguments are positional and explicit. `Workspace(shape, dtype="uint8",
initialization="zero", alignment=1)` adds a private workspace pointer; choose
`uninitialized` only if the program initializes everything it reads. Grid, block
and dynamic shared memory remain mutable per Candidate. Input/output aliasing is
not provided. A declaration does not enforce read-only inputs; request the check.

The worker assembles for its device architecture with ptxas, loads the cubin with
the CUDA driver, and launches on Torch's current stream. Torch owns allocations.
PTX, cubin, hashes, assembler version/options/log, launch and queried function
resources are retained even without profiling. Alignment violations are reported.
Output/workspace allocation and workspace zeroing are inside the call; input
restoration is outside timing. Cache state is uncontrolled. Do not infer FP32
accumulation merely from BF16 input/output declarations.

The implementation follows the NVIDIA [driver launch API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
and [function resource attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html).
CPU assembly is qualified; GPU load/launch and memory checks still require a
compatible live GPU. No PTX runtime speedup is claimed.

## Vendor baselines

```python
from gpu2tensor.vendor import matmul

baseline = worker.evaluate(matmul("trainium"), workload, repetitions=30)
# Use matmul("cuda") on a CUDA worker.
```

The factory returns an ordinary Candidate with explicit Torch source and language
`torch`. CUDA preserves Torch matmul defaults and records the effective precision flags.
Use `matmul("cuda", precision="strict")` for a separately named variant disabling
TF32 and BF16 reduced-precision reduction. Both must pass the same client
validator; report a default-baseline contract failure explicitly instead of
substituting the strict variant without labeling it. Trainium compiles through Torch-Neuron and uses the same resident NRT
runner and timer as NKI. The adapter does not time Torch's host input/output
transfers against a resident NKI kernel. Compilation runs in a per-job directory.

Compare `backend`, `host`, `device`, `neuron_core` (or CUDA device identity),
`inputs` (shape/dtype/hash), `benchmark_case`, `timing_method`, and
`measurement_policy` before comparing latency samples. The policy includes
warmup count, input restoration, output allocation/reuse, synchronization, working
set and cache treatment. Cache state is uncontrolled; a matching policy does not
mean identical cache contents. CUDA output allocation is inside the call; Neuron
outputs are resident and reused, for both the baseline and candidate. These are
same-backend comparisons, not cross-device speedups.

Neuron `programs` includes NEFF hashes, Torch/compiler/dependency versions,
compiler arguments and precision declarations. HLO, ABI metadata, NEFF and compiler
logs are retained without profiling. Auto-casting is disabled; accumulation is
reported as vendor-default and must pass the client validator. Do not infer a
precision guarantee from a compiler flag alone. CUDA records effective matmul
precision flags. A baseline tag applies only to the factory's exact source.

The operator sets `GPU2TENSOR_TORCH_NEURON_PYTHON` on the worker to a compatible
compiler environment with gpu2tensor installed. The package installs no SDK.
The adapter uses Torch-Neuron's trace frontend before its TorchScript runtime
wrapper to obtain a NEFF and tensor ABI. That extraction API is version-sensitive;
see [qualified environments](environments.md). The vendor's
[trace documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-trace.html)
describes the compilation and artifact work directory.
