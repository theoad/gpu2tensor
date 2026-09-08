# Inspect lowered instructions

This optional collection path is implemented and tested with local fake tools.
Real CUDA disassembly and Trainium timeline export remain **unqualified**. It
does not establish that a proposed operation order survives compilation.

Submit a complete PTX candidate and its workload as usual:

```python
result = worker.evaluate(ptx_candidate, workload, artifacts=("disassembly",))
info = result.record.get("instruction_artifacts", {}).get("disassembly")
if info and info["status"] == "ok":
    result.save("artifacts/schedule-a")
```

For NKI or Torch on Trainium, request the separate profiled execution explicitly:

```python
result = worker.evaluate(nki_candidate, workload, profile=True,
                         artifacts=("instruction_timeline",))
print(result.record.get("instruction_artifacts", {}))
```

`Evaluator.observe`, `Pool.observe` and `KernelEnv` accept the same `artifacts`
option. Empty options preserve request version 2 and current behavior. Nonempty
options use version 3, which older workers reject. Missing artifact metadata in
an old result or a failed worker response provides no inspection evidence.
`instruction_timeline` requires `profile=True`; it never silently adds a run.

Each requested entry reports `status`, `reason`, `implemented`, `qualification`,
scope, input hashes, tool path/hash/version, command and retained file hashes.
`implemented` describes the adapter path, not hardware qualification. Statuses:

| Status | Meaning |
| --- | --- |
| `ok` | The decoder succeeded and the expected output was retained. Qualification remains separate. |
| `unavailable` | Backend/language unsupported, native artifact missing, or operator tool missing. |
| `not_collected` | Evaluation did not reach inspection, for example because correctness failed. |
| `failed` | The tool failed, timed out, exceeded a limit or returned an invalid output envelope. |

Inspection occurs after benchmarking and profiling. Inspection failure preserves
the correctness outcome and timing samples. It is not an invalid-program label.
`instruction_artifact_seconds` measures inspection wall time, not kernel time.

## CUDA

PTX candidates retain the exact cubin independently of this option. An
operator-provided `nvdisasm` is preferred, with `cuobjdump -sass` used when
nvdisasm is absent. A failed nvdisasm run is reported rather than silently using
a different decoder. Both tools must understand the cubin architecture.

The extra files are `instructions/disassembly/sass.txt` and `decoder.log`.
They describe **static lowered instructions**, not dynamic warp issue order.
Triton and Torch disassembly are unavailable until their actual executed
binaries can be retained and associated with each launch. No compiler flags or
candidate source are changed to improve the listing.

## Trainium

An operator-provided `neuron-explorer` consumes the captured NEFF/NTFF. The
adapter requests Parquet export into a temporary directory and retains these
tables, including every numbered chunk it finds:

- `Instruction`, `BirInstruction`
- `DmaPacket`, `DmaPacketAggregated`, `SemaphoreUpdate`
- `SchemaFields`

An Instruction table is required. `missing_tables` lists other absent tables;
absence does not imply zero events. The adapter checks the file envelope and
preserves native bytes without loading Parquet libraries or changing intervals.
It does not validate the vendor schema or prove trace completeness. Clients may
use their own Parquet reader after saving the result. Concurrent engine intervals
remain concurrent; the package does not create a global instruction order.

`source_mapping_status="not_checked"`, `completeness="not_verified"` and
`dropped_events=null` are deliberate. The client must inspect mapping columns
and the retained schema before making a schedule-survival claim. Existing NEFFs
may lack source/debug mappings; this change does not enable debug compilation or
retain temporary compiler IR. Trn1 scheduling controls remain unqualified even
when a timeline can be decoded.

## Limits and ownership

Tools are found on the operator's PATH. The package installs nothing. Decoder
version checks have a 10-second/16-KiB output limit; decoding has a 60-second,
64-MiB output limit and at most 128 filesystem entries. Stderr is limited to
1 MiB. Polling limits allow transient disk overshoot between checks; they are
resource controls for trusted tools, not an OS sandbox. Returned artifacts also
must fit the existing 256-MiB expanded result limit, including request copies.
Oversized or partial inspection output is discarded with an explicit failure.

The current worker job deadline still includes inspection. Choose an appropriate
operator deadline for a profiled evaluation. Candidate compilation and execution
remain separate from these decoder commands.

## Proposed offline qualification commands

The [offline CUDA tool procedure](offline-cuda-tools.md) supplies exact official
12.8.1 archive URLs, checksums and task-local extraction commands for a Linux
x86-64 host with no toolkit installed.

These commands are proposed only; no toolkit has been installed and no real
decoder has been run for this change. Use an existing authorized Linux host with
a compatible CUDA Binary Utilities installation or Neuron Explorer (the archived
Trainium profiles identify Explorer `2.32.0.498-b1f3998`). First inspect the
installed tool version. Point IN at a copy of an archived result and OUT at a
new directory; never overwrite the original. No candidate source is executed.

```bash
timeout 10s nvdisasm --version
timeout 60s nvdisasm "$IN/candidate.cubin" > "$OUT/sass.txt" 2> "$OUT/decoder.log"

timeout 10s neuron-explorer --version
timeout 60s neuron-explorer view -n "$IN/kernel.neff" -s "$IN/kernel.ntff" \
  --output-format parquet --output-file "$OUT/tables" --data-path "$OUT/state" \
  --disable-ui --ingest-only
```

Run under the same disk/file limits as the adapter, then retain tool versions,
commands and all input/output hashes. The exact installed Explorer flags and
export layout still require qualification. If a flag is unsupported, preserve
the diagnostic and stop; do not install a different tool or launch a worker.

References: [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html),
[Explorer export](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/how-to-analyze-profile-output.html),
[Explorer schema](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/profile-schema-reference.html).
