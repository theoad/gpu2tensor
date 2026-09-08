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
