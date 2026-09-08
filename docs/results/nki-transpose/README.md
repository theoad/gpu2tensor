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

Hardware qualification passed on replacement Trainium1 host `ip-172-31-69-68`,
core 0, at runtime revision `6ebf286`: the exact failed request returned
`candidate_compile` with this policy and no timing. The saved valid 128³ BF16
request passed all six input families and buffer diagnostics; its six captured
output files were byte-identical to the original. Both requests retained their
source/input/request hashes. See [report](report.json) and
[the immutable saved classification](saved-classification.json).

Only `failures.py` and the Trainium diagnostic-selector import/call differ in
returned engine hashes from the original runtime. Normalizing that selector name
makes the Trainium adapter byte-identical to `ae03dbc`. All successful execution,
validation, timing and profiling paths are unchanged. Four focused policy tests
and all 47 integrated tests pass.

AWS reclaimed the original Spot runner for lack of capacity at 15:24 UTC on
2026-09-08. Its patch command was undeliverable and ran no controls. The replacement
uses the same pinned AMI and SDK, with a new host identity; these checks are not
a continuation of the old host's performance experiment. All 65 files across both
original result directories stayed unchanged. Raw results and interruption
records are under ignored `artifacts/issue9/` and the operator S3 `issue9/results/`
prefix. The experiment owns any new baseline and search condition.

The separate vendor compiler environment was restored at its recorded versions.
[A vendor 128³ BF16 control](vendor-control.json) also passed all six cases on the
replacement. This checks dependency readiness; the experiment still collects a
new matched baseline for its new seed/host condition.
