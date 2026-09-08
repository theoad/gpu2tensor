# NKI frontend classification

Policy `nki-0.6-buffer-placement-v1` recognizes two confirmed NKI 0.6 parser
assertions: `nc_transpose data` and `tensor_copy dst` placed in `shared_hbm` when
SBUF/PSUM is required. The adapter wraps only an `AssertionError` from the frontend
IR compilation call, and only when every collected diagnostic matches that policy.
Internal compiler errors, other assertions, missing SDKs, runtime errors and
timeouts remain unknown. This is deliberately narrower than all compiler errors.

On 2026-09-08, Trainium `ip-172-31-64-161`, physical core 0, NKI
0.6.0+31049202112.g85070674, both exact failed experiment requests were replayed.
Both returned `failure_kind="candidate_compile"`, a structured
`compiler_diagnostic`, original source/input identity and no timing. The valid
BF16 NKI control still passed. See [report](report.json). Two focused local tests
cover strict matching and immutable legacy classification. AST comparison against
8924b21 confirms `benchmark`, `run`, `resident`, `identity`, `profile`, and `check`
are unchanged. The only runtime edits wrap frontend compilation and classify the
typed exception; timing and wire contracts are unchanged.

## Existing immutable records

```python
from gpu2tensor.failures import classify_saved_nki_failure

annotation = classify_saved_nki_failure(original_record, original_process_log)
# None means unknown. Save a returned annotation alongside the original evidence.
```

Legacy classification additionally requires the NKI frontend traceback and the
exact matching terminal AssertionError. The returned new dictionary includes
`failure_kind`, `diagnostic`, `canonical_record_sha256` (sorted compact JSON), and
`process_log_sha256` (the supplied text encoded as UTF-8). It never changes the
record. Keep the original file-byte hashes separately if required by your ledger.
This is a policy annotation, not a new hardware evaluation or numerical result.
