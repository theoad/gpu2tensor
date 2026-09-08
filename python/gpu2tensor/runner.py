"""Run one candidate in a child process, with a bounded execution deadline."""

import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import zipfile

from gpu2tensor.client import MAX_RESULT_BYTES

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def evaluate_archive(payload, backend, timeout=300, *, process=None):
    if backend not in {"cuda", "trainium", "cpu"}:
        raise ValueError("Unknown backend.")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Request exceeds 64 MiB.")
    with tempfile.TemporaryDirectory(prefix="gpu2tensor-") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_REQUEST_BYTES:
                raise ValueError("Expanded request exceeds 64 MiB.")
            for item in archive.infolist():
                path = Path(item.filename)
                if path.is_absolute() or ".." in path.parts or item.is_dir():
                    raise ValueError("Invalid request path.")
                target = directory / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
        output = directory / "output"
        output.mkdir()
        if process is not None:
            failure = process.run(directory, backend, timeout)
        else:
            with (output / "process.log").open("wb") as log:
                child = subprocess.Popen(
                    [sys.executable, "-m", "gpu2tensor.execute", str(directory), backend],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                failure = None
                try:
                    code = child.wait(timeout=timeout)
                    if code:
                        failure = {"status": "worker_error", "message": f"Candidate process exited with code {code}."}
                except subprocess.TimeoutExpired:
                    failure = {"status": "timeout", "message": f"Candidate exceeded {timeout} seconds."}
                finally:
                    # Include descendants such as compiler processes in cleanup.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
        if failure:
            failure.update({"version": 1, "backend": backend})
            (output / "record.json").write_text(json.dumps(failure))
        if not (output / "record.json").exists():
            raise RuntimeError("Candidate process returned no result.")
        for name in ("candidate.py", "reference.py", "request.json"):
            (output / name).write_bytes((directory / name).read_bytes())
        if (directory / "validator.py").is_file():
            (output / "validator.py").write_bytes((directory / "validator.py").read_bytes())
        # Retain exact inputs and source so an interrupted run can be replayed.
        (output / "request.zip").write_bytes(payload)
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > MAX_RESULT_BYTES:
            raise RuntimeError("Result exceeds 256 MiB; reduce the requested profile scope.")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                if path.is_file():
                    archive.write(path, str(path.relative_to(output)))
        return buffer.getvalue()
