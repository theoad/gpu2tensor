# Evaluation contract

A candidate contains complete source and its language. A workload contains a
reference module with `reference(*inputs)` and independent, C-contiguous NumPy
input cases. Strided tensor contracts and multiple outputs are deferred.
The first adapters accept Python modules: Triton exports `run(*torch_inputs)`;
NKI exports an undecorated `kernel(*numpy_inputs)` that the backend compiles.
Other IR languages are a future adapter, not accepted silently by v0.

An evaluator runs correctness cases, then benchmarks case zero after warmup,
then optionally profiles a separate execution. The result keeps timing samples,
diagnostics, input/source hashes, versions, and raw vendor artifacts. It never
benchmarks a candidate that fails correctness. Float tolerances and finite-output
checks are empirical checks on supplied cases, not a proof of equivalence.

`observe` yields one completed evaluation at a time without Gym or a policy.
The optional Gym wrapper submits complete source strings and adds client-owned
reward/episode rules. A worker handles one request at a time on one device.
Multiple workers can be addressed independently; a distributed scheduler is not
part of this first slice.

Requests use ZIP containers with JSON metadata and NumPy arrays (never pickle).
Remote transport currently buffers one bounded request and one result, not an
unbounded stream. Vendor profile parsing and remote transfer require copies.
Owned contiguous NumPy feature arrays can share CPU storage with Torch through
`from_numpy`; device transfer is explicit and owns its destination. No end-to-end
zero-copy claim applies to this version.

The worker executes trusted candidate/reference code in a child process. This
separates crashes and ordinary timeouts, but is not a security sandbox. Keep the
loopback endpoint behind an operator-owned tunnel. Compiler descendants are
killed on timeout. A device fault may require restarting/replacing the worker.

Profile availability, sampling, replay and completeness are separate metadata.
CUDA event timings include host launch gaps for small kernels; they are not
isolated instruction latency. We do not equate vendor counter names or fill
unavailable counters with zeros. Replayed/instrumented durations do not replace
the uninstrumented benchmark label.

Trainium v0 is qualified against NKI 0.6 and the pinned Neuron image. It captures
NTFF through the runtime, then optionally calls Neuron Explorer to decode a native
summary. `profile.measurements` keeps vendor counter names and explicit byte/count
units. Decoder absence is `summary_status=tool_unavailable`; raw trace capture can
still succeed. GPU profiling currently returns kernel launch aggregates through
Proton/CUPTI. Both report unverified completeness and unknown dropped events.

Operators may opt into [bounded process reuse and PTX caching](process-reuse.md).
Fresh processes remain the default; the client and observation-only API stay the same.
