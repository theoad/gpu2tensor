"""Optional offline decoding of binaries and traces by operator-provided tools."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


KINDS = ("disassembly", "instruction_timeline")
MAX_BYTES = 64 * 1024 * 1024
MAX_FILES = 128
MAX_LOG_BYTES = 1024 * 1024
TABLE = re.compile(r"(Instruction|BirInstruction|DmaPacket|DmaPacketAggregated|SemaphoreUpdate|SchemaFields)(?:_[0-9]+)?\.parquet")


def requested(kinds, profile):
    """Reject misspelled options and unintended extra profiling executions."""
    if isinstance(kinds, str):
        raise ValueError("Pass artifact names as a list or tuple.")
    kinds = tuple(kinds)
    if any(kind not in KINDS for kind in kinds) or len(set(kinds)) != len(kinds):
        raise ValueError(f"Choose unique artifact names from {KINDS}.")
    if "instruction_timeline" in kinds and not profile:
        raise ValueError("instruction_timeline requires profile=True for its separate execution.")
    return kinds


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def files(directory):
    """Decoders own a temporary directory; never follow links out of it."""
    paths = []
    for path in directory.rglob("*"):
        paths.append(path)
        if len(paths) > MAX_FILES:
            raise ValueError("Decoder output exceeds the file limit.")
    if any(path.is_symlink() for path in paths):
        raise ValueError("Decoder output contains a symbolic link.")
    result = [path for path in paths if path.is_file()]
    return result


def run_tool(command, directory, *, timeout=60, max_bytes=MAX_BYTES):
    """Bound decoder time and disk output, without buffering its stdout in RAM."""
    stdout, stderr = directory / "stdout.txt", directory / "stderr.txt"
    started = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, cwd=directory)
        try:
            while True:
                paths = files(directory)
                if sum(path.stat().st_size for path in paths) > max_bytes:
                    raise ValueError("Decoder output exceeds the byte limit.")
                if stderr.stat().st_size > MAX_LOG_BYTES:
                    raise ValueError("Decoder log exceeds the byte limit.")
                code = child.poll()
                if code is not None:
                    break
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Decoder exceeded its time limit.")
                time.sleep(0.05)
            if sum(path.stat().st_size for path in files(directory)) > max_bytes:
                raise ValueError("Decoder output exceeds the byte limit.")
            if stderr.stat().st_size > MAX_LOG_BYTES:
                raise ValueError("Decoder log exceeds the byte limit.")
            if code:
                raise RuntimeError(f"Decoder exited with code {code}: {stderr.read_text(errors='replace')[-2000:]}")
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
    return stdout, stderr


def capability(kind, backend, language):
    supported = ((kind == "disassembly" and backend == "cuda" and language == "ptx") or
                 (kind == "instruction_timeline" and backend == "trainium" and language in ("nki", "torch")))
    return {"status": "not_collected" if supported else "unavailable",
            "reason": "evaluation_not_completed" if supported else "backend_or_language_not_supported",
            "implemented": supported, "qualification": "unqualified",
            "scope": "static_lowered_instructions" if kind == "disassembly" else "profiled_engine_intervals",
            "completeness": "not_verified", "dropped_events": None,
            "source_mapping_status": "not_checked", "files": []}


def collect(output, backend, language, kinds):
    """Decode after timing. Inspection failure never changes correctness or latency."""
    results = {}
    for kind in kinds:
        info = results[kind] = capability(kind, backend, language)
        if not info["implemented"]:
            continue
        inputs = ["candidate.cubin"] if kind == "disassembly" else ["kernel.neff", "kernel.ntff"]
        if not all((output / name).is_file() for name in inputs):
            info.update(status="unavailable", reason="native_artifact_missing")
            continue
        names = ("nvdisasm", "cuobjdump") if kind == "disassembly" else ("neuron-explorer",)
        tool = next((path for name in names if (path := shutil.which(name))), None)
        if tool is None:
            info.update(status="unavailable", reason="operator_tool_missing", tools=list(names))
            continue
        try:
            info["inputs"] = {name: digest(output / name) for name in inputs}
            tool = os.path.abspath(tool)
            info["tool"] = {"path": tool, "sha256": digest(tool)}
            with tempfile.TemporaryDirectory(prefix="gpu2tensor-inspect-") as temporary:
                directory = Path(temporary)
                version_dir = directory / "version"
                version_dir.mkdir()
                version_command = [tool, "--version"]
                out, err = run_tool(version_command, version_dir, timeout=10, max_bytes=16384)
                info["tool"].update(version=(out.read_text(errors="replace") + err.read_text(errors="replace")).strip(),
                                    version_command=version_command)
                decode_dir = directory / "decode"
                decode_dir.mkdir()
                if kind == "disassembly":
                    command = [tool, *(["-sass"] if Path(tool).name == "cuobjdump" else []), str((output / inputs[0]).resolve())]
                else:
                    command = [tool, "view", "-n", str((output / inputs[0]).resolve()),
                               "-s", str((output / inputs[1]).resolve()), "--output-format", "parquet",
                               "--output-file", str(decode_dir / "tables"),
                               "--data-path", str(decode_dir / "state"), "--disable-ui", "--ingest-only"]
                info["command"] = command
                out, err = run_tool(command, decode_dir)
                if kind == "disassembly":
                    if not out.stat().st_size:
                        raise ValueError("Decoder returned an empty disassembly.")
                    selected = [(out, "sass.txt"), (err, "decoder.log")]
                else:
                    # The data-path may contain a second cached copy. Return
                    # only the requested export so consumers cannot count it twice.
                    selected = [(path, str(path.relative_to(decode_dir)))
                                for path in files(decode_dir / "tables") if TABLE.fullmatch(path.name)]
                    for path, _ in selected:
                        with path.open("rb") as stream:
                            if path.stat().st_size < 12 or stream.read(4) != b"PAR1":
                                raise ValueError("Decoder returned an invalid Parquet file.")
                            stream.seek(-4, 2)
                            if stream.read(4) != b"PAR1":
                                raise ValueError("Decoder returned an invalid Parquet file.")
                    present = sorted({TABLE.fullmatch(path.name).group(1) for path, _ in selected})
                    if "Instruction" not in present:
                        raise ValueError("Decoder returned no Instruction table.")
                    info["tables"] = present
                    info["missing_tables"] = sorted(set(("Instruction", "BirInstruction", "DmaPacket", "DmaPacketAggregated", "SemaphoreUpdate", "SchemaFields")) - set(present))
                    selected += [(out, "decoder.stdout.txt"), (err, "decoder.log")]
                # Reserve space for the final record and the runner's request copies.
                from gpu2tensor.client import MAX_RESULT_BYTES
                existing = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                request_bytes = sum(path.stat().st_size for path in output.parent.rglob("*") if path.is_file() and output not in path.parents)
                if existing + 2 * request_bytes + sum(path.stat().st_size for path, _ in selected) + MAX_LOG_BYTES > MAX_RESULT_BYTES:
                    raise ValueError("Instruction artifacts would exceed the result byte limit.")
                destination = output / "instructions" / kind
                for path, name in selected:
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    info["files"].append({"file": str(target.relative_to(output)), "bytes": target.stat().st_size, "sha256": digest(target)})
                info.update(status="ok", reason=None)
        except (OSError, ValueError, RuntimeError) as error:
            shutil.rmtree(output / "instructions" / kind, ignore_errors=True)
            info.update(status="failed", reason=str(error), files=[])
    return results
