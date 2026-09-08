"""Narrow, versioned policies for confirmed candidate compiler diagnostics."""

import hashlib
import json
import re

NKI_BUFFER_POLICY = "nki-0.6-buffer-placement-v1"
_BUFFER = re.compile(r"- \[x[1-9][0-9]*\] error: assertion failed: (?:nc_transpose data|tensor_copy dst) must be in \[(?:sbuf|psum)(?:, (?:sbuf|psum))*\], got shared_hbm")


class CandidateCompilationError(ValueError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def nki_buffer_diagnostic(message):
    lines = message.strip().splitlines()
    if len(lines) < 3 or lines[0] != "error: failed to compile NKI kernel:":
        return None
    count = re.fullmatch(r"Collected ([1-9][0-9]*) different diagnostics:", lines[1])
    if count is None or int(count[1]) != len(lines) - 2:
        return None
    if not all(_BUFFER.fullmatch(line) for line in lines[2:]):
        return None
    return {"policy": NKI_BUFFER_POLICY, "provider": "nki_frontend",
            "exception_type": "AssertionError", "messages": lines[2:]}


def classify_saved_nki_failure(record, process_log):
    """Return a separate classification; never rewrite the original record.

    Legacy records lack a typed compiler field. Require the qualified frontend
    traceback and the exact terminal assertion as well as the known diagnostic.
    A missing match leaves the outcome unknown.
    """
    diagnostic = nki_buffer_diagnostic(record.get("message", ""))
    if (record.get("backend") != "trainium" or record.get("language") != "nki"
            or record.get("status") != "error" or not record.get("nki", "").startswith("0.6.")
            or record.get("stage") not in ("correctness", "prepare") or record.get("latency_ms")
            or diagnostic is None):
        return None
    if ('nki/compiler/frontend.py' not in process_log or 'compile_kernel_to_nir' not in process_log
            or not process_log.rstrip().endswith('AssertionError: ' + record['message'].rstrip())):
        return None
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"failure_kind": "candidate_compile", "diagnostic": diagnostic,
            "canonical_record_sha256": hashlib.sha256(canonical).hexdigest(),
            "process_log_sha256": hashlib.sha256(process_log.encode()).hexdigest()}
