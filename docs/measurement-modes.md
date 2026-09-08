# Measurement modes: proposed public contract

This is design work for issues #10 and #14. No new timing mode is implemented or
qualified here, and the frozen campaign keeps its existing client and worker.

Today `evaluate(..., repetitions=30)` returns raw samples in
`result.record["latency_ms"]`; `Result.latency_ms` is their median. CUDA uses
`cuda_events_including_launch_gaps`; Trainium uses
`neuron_host_round_trip_with_resident_tensors`. Both exclude compilation from
the sampled interval. They do not isolate device execution from all frontend
or launch overhead. Profiles are separate executions.

## Proposed client option

Add `timing="default"` to evaluation, observation and Gym when there is a
qualified implementation. Proposed explicit choices are `cuda_graph_replay`
and `neuron_device`. Never silently fall back from a requested mode. A worker
must report `unavailable` with a reason such as `backend_not_supported`,
`toolchain_not_qualified` or `capture_not_supported`. No timing samples should
appear under an unavailable requested mode, even if correctness succeeded.

Device timing is a capability of the full language/compiler/runtime/device
combination. Candidate and official vendor baseline must use the same policy;
qualifying NKI does not automatically qualify Torch-Neuron.

## Proposed result fields

Keep the existing default fields compatible. New modes should additionally
return a `measurement` object with this information:

| Field | Contract |
| --- | --- |
| `requested_mode`, `method`, `status`, `reason` | Requested policy, actual boundary and explicit availability. |
| `unit` | `ms`, after a documented conversion from the provider's original unit. |
| `samples` | Raw samples when supplied; null when the provider supplies only aggregates. |
| `statistics` | Provider mean/min/max/std, sample count and statistic definitions, or null. Never fabricate samples from summaries. |
| `policy` | Warmup count, timed count, replay count, allocation, input restoration/residency, output/workspace reuse, cache, synchronization, profiled/instrumented status. |
| `qualification` | Exact device, runtime/compiler versions and the qualification evidence identifier. |

Aggregate-only results must not populate legacy `latency_ms` with a made-up
singleton mean. The existing median convenience property should return None
when there are no raw samples. The client chooses its reward statistic and
must explicitly consume aggregate-only results. This avoids silently changing
the meaning of existing rewards or replay data.

Return separately attributed wall durations for compilation, model loading,
correctness, benchmark orchestration and profile/inspection work. Missing phase
measurements are null, not zero. The current `prepare_and_check_seconds` is a
combined legacy measurement and cannot supply this breakdown retroactively.
Device samples remain distinct from these wall durations. No estimated overhead
subtraction and no relabeling of a profiled duration as an unprofiled sample.

## Qualification gates

CUDA graph replay needs preallocated input/output/workspace buffers,
preinitialized timing events and an explicit restoration schedule. Test PTX and
the Torch baseline with identical working sets, cache policy and replay count.
Validate outputs and mutation controls before and after replay. Introduce
controlled host launch delays to establish which overhead the interval excludes.
Unsupported allocation, stream or capture behavior must reject the mode.

The matching NKI 0.6 source has a CompiledKernel benchmark path calling
`model.benchmark(mode="device", warmup_iter=..., benchmark_iter=...)`. Its
runtime shim is internal and returns aggregate statistics at that layer. Before
exposing a mode, qualify Trn1 support, actual timing boundaries/units,
per-iteration or batch semantics, input restoration and raw sample availability.
Do not infer those properties from the symbol's presence. Device tracing, if
used, must be named as instrumented rather than described as unprofiled.

For schedule survival, use two fixed complete programs with a legal independent
operation reordering and mapped operation identities. Repeat compilation to
check lowering stability; validate outputs; inspect actual lowered dependencies
or per-engine intervals. Binary hashes and instruction counts alone are
insufficient. Missing mapping is inconclusive. Then measure balanced A/B and B/A
blocks on one reserved device. Keep hosts, sessions and old campaign blocks
separate. The client owns this pairing and interpretation; no pair or trajectory
protocol is needed.
