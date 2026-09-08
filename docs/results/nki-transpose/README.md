# NKI vector-transpose diagnostic

Issue #9 adds policy `nki-0.6-vector-transpose-shape-v1` for the observed NKI 0.6
frontend assertion: `Vector engine transpose requires shape <= [32, 32], got
[64, 32]`. The exact saved compiler log points to the NKI ISA shape validator,
called while tracing the candidate. Other shapes, limits, diagnostics and mixed
unrecognized diagnostic groups remain unknown under this policy.

`nki_compile_diagnostic` tries the existing buffer-placement policy first, then
this new policy. `nki_buffer_diagnostic` keeps its previous matching behavior and
policy name. The Trainium adapter changes only its diagnostic selector inside
the existing frontend compilation exception handler. Successful execution,
validation, timing and profiling methods are unchanged.

`classify_saved_nki_failure(record, process_log)` retains its signature and returns
a separate annotation or `None`. It requires NKI 0.6, an error before timing, the
frontend compilation traceback and an exact terminal assertion. The annotation
includes the policy and record/log hashes; original evidence is never rewritten.
The original issue #7 behavior remains available at its pinned `ae03dbc` revision.

Four focused local tests pass, covering the old policy, the new exact diagnostic,
unknown cases and immutable saved-record handling. The actual saved issue #9
record/log also match. These are classification checks, not a new hardware result.

Hardware qualification is pending: AWS reclaimed the original Spot runner for
lack of capacity at 15:24 UTC on 2026-09-08. The deployment command was undeliverable;
no patch was installed and neither requested control ran. The original request,
record and logs remain immutable. The required next checks are the exact failed
request and a saved valid 128x128x128 BF16 request covering all six input families.
Do not describe this policy as hardware-qualified until those checks succeed.
