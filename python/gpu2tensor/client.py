"""Synchronous evaluation and observation-only collection."""

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

MAX_RESULT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    source: str
    language: str
    name: str = "candidate"

    @classmethod
    def from_file(cls, path, language):
        path = Path(path)
        return cls(path.read_text(), language, path.stem)


@dataclass
class Workload:
    """A reference module defining reference(*inputs), and independent test cases."""

    reference: str
    cases: list[tuple[np.ndarray, ...]]
    rtol: float = 1e-4
    atol: float = 1e-5

    def __post_init__(self):
        if not self.cases or not all(case for case in self.cases):
            raise ValueError("Provide at least one nonempty input case.")
        for case in self.cases:
            for array in case:
                if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
                    raise TypeError("Inputs must be real numeric NumPy arrays.")
                if not array.flags.c_contiguous:
                    raise ValueError("The v0 input contract requires C-contiguous arrays.")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("Correctness tolerances cannot be negative.")


@dataclass
class Result:
    record: dict
    artifacts: dict[str, bytes] = field(default_factory=dict)

    @property
    def correct(self):
        return self.record["status"] == "ok"

    @property
    def latency_ms(self):
        samples = self.record.get("latency_ms", [])
        return float(np.median(samples)) if samples else None

    def save(self, directory):
        """Persist the complete result before reusing an interruptible worker."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError("Choose an empty result directory to keep artifacts from different runs separate.")
        for name, data in self.artifacts.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temporary = directory / "record.json.tmp"
        temporary.write_text(json.dumps(self.record, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "record.json")

    @classmethod
    def load(cls, directory):
        return cls(json.loads((Path(directory) / "record.json").read_text()))


def pack_request(candidate, workload, profile, repetitions):
    manifest = {"version": 1, "language": candidate.language, "name": candidate.name,
                "case_sizes": [len(case) for case in workload.cases],
                "rtol": workload.rtol, "atol": workload.atol,
                "profile": profile, "repetitions": repetitions}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("request.json", json.dumps(manifest, allow_nan=False))
        archive.writestr("candidate.py", candidate.source)
        archive.writestr("reference.py", workload.reference)
        for case_index, case in enumerate(workload.cases):
            for input_index, array in enumerate(case):
                array_buffer = io.BytesIO()
                np.save(array_buffer, array, allow_pickle=False)
                archive.writestr(f"inputs/{case_index}-{input_index}.npy", array_buffer.getvalue())
    return buffer.getvalue()


def unpack_result(data):
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("Result exceeds 256 MiB.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_RESULT_BYTES:
            raise ValueError("Expanded result exceeds 256 MiB.")
        artifacts = {}
        for item in archive.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid artifact path in worker result.")
            if not item.is_dir() and item.filename != "record.json":
                artifacts[item.filename] = archive.read(item)
        return Result(json.loads(archive.read("record.json")), artifacts)


class Evaluator:
    """Use a local backend or an operator-provided, forwarded worker endpoint."""

    def __init__(self, endpoint=None, *, backend=None, timeout=300):
        if (endpoint is None) == (backend is None):
            raise ValueError("Choose either endpoint= or backend=.")
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.backend = backend
        self.timeout = timeout

    def evaluate(self, candidate, workload, *, profile=False, repetitions=30):
        if not isinstance(repetitions, int) or not 1 <= repetitions <= 10000:
            raise ValueError("repetitions must be between 1 and 10000.")
        payload = pack_request(candidate, workload, profile, repetitions)
        if self.endpoint:
            request = urllib.request.Request(self.endpoint + "/evaluate", data=payload,
                                             headers={"Content-Type": "application/zip"})
            with urllib.request.urlopen(request, timeout=self.timeout + 30) as response:
                data = response.read(MAX_RESULT_BYTES + 1)
        else:
            from gpu2tensor.runner import evaluate_archive
            data = evaluate_archive(payload, self.backend, self.timeout)
        return unpack_result(data)

    def observe(self, candidates, workload, *, profile=True, repetitions=30):
        """Yield each completed result; never buffer the entire candidate corpus."""
        for candidate in candidates:
            yield self.evaluate(candidate, workload, profile=profile, repetitions=repetitions)
