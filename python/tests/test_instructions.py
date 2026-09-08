"""Contract tests use tiny fake decoder executables, never accelerator code."""

import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

from gpu2tensor import Evaluator
from gpu2tensor.client import pack_request
from gpu2tensor.examples.softmax import candidate, workload
from gpu2tensor import instructions


def decoder(tmp_path, monkeypatch, name, body):
    tool = tmp_path / name
    tool.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                    "if '--version' in sys.argv:\n    print('mock decoder 1.0')\n    sys.exit(0)\n" + body)
    tool.chmod(0o755)
    monkeypatch.setattr(instructions.shutil, "which", lambda wanted: str(tool) if wanted == name else None)
    return tool


def native(tmp_path, *names):
    output = tmp_path / "output"
    output.mkdir()
    for name in names:
        (output / name).write_bytes(b"native input")
    return output


def test_opt_in_protocol_and_default_compatibility():
    def manifest(artifacts=(), profile=False):
        payload = pack_request(candidate("cpu"), workload(), profile, 1, artifacts)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return json.loads(archive.read("request.json"))
    assert manifest()["version"] == 2
    assert "artifacts" not in manifest()
    assert manifest(("disassembly",))["version"] == 3
    assert manifest(("instruction_timeline",), True)["artifacts"] == ["instruction_timeline"]
    for invalid in ("disassembly", ["typo"], ["disassembly", "disassembly"]):
        with pytest.raises(ValueError):
            manifest(invalid)
    with pytest.raises(ValueError, match="profile=True"):
        manifest(["instruction_timeline"])


def test_unsupported_artifact_keeps_cpu_correctness_and_samples(tmp_path):
    result = Evaluator(backend="cpu").evaluate(candidate("cpu"), workload(), repetitions=2,
                                               artifacts=("disassembly",))
    assert result.correct and len(result.record["latency_ms"]) == 2
    info = result.record["instruction_artifacts"]["disassembly"]
    assert info["status"] == "unavailable" and not info["implemented"]
    result.save(tmp_path / "saved")
    assert not (tmp_path / "saved/instructions").exists()


def test_version_three_preserves_bfloat16_input_identity():
    import torch
    from gpu2tensor import Candidate, Workload
    source = Candidate("def run(x):\n    return x.clone()\n", "python")
    target = Workload("def reference(x):\n    return x.clone()\n",
                      [(torch.ones(4, dtype=torch.bfloat16),)])
    result = Evaluator(backend="cpu").evaluate(source, target, repetitions=1, artifacts=("disassembly",))
    assert result.correct and result.record["inputs"][0][0]["dtype"] == "bfloat16"


@pytest.mark.parametrize("name", ["nvdisasm", "cuobjdump"])
def test_disassembly_provenance_and_exact_input(tmp_path, monkeypatch, name):
    output = native(tmp_path, "candidate.cubin")
    tool = decoder(tmp_path, monkeypatch, name, "print('/*0000*/ FADD R0, R1, R2;')\n")
    before = instructions.digest(output / "candidate.cubin")
    info = instructions.collect(output, "cuda", "ptx", ["disassembly"])["disassembly"]
    assert info["status"] == "ok" and info["qualification"] == "unqualified"
    assert info["scope"] == "static_lowered_instructions"
    assert info["inputs"] == {"candidate.cubin": before}
    assert info["tool"]["sha256"] == instructions.digest(tool)
    assert info["tool"]["version"] == "mock decoder 1.0"
    assert ("-sass" in info["command"]) == (name == "cuobjdump")
    assert info["command"][-1] == str((output / "candidate.cubin").resolve())
    assert (output / info["files"][0]["file"]).read_text() == "/*0000*/ FADD R0, R1, R2;\n"
    assert instructions.digest(output / "candidate.cubin") == before


def test_missing_tool_and_missing_native_artifact(tmp_path, monkeypatch):
    output = native(tmp_path, "candidate.cubin")
    monkeypatch.setattr(instructions.shutil, "which", lambda _: None)
    assert instructions.collect(output, "cuda", "ptx", ["disassembly"])["disassembly"]["reason"] == "operator_tool_missing"
    assert instructions.collect(output, "trainium", "nki", ["instruction_timeline"])["instruction_timeline"]["reason"] == "native_artifact_missing"


@pytest.mark.parametrize("body,reason", [("sys.exit(4)\n", "code 4"), ("pass\n", "empty disassembly"),
    ("sys.stderr.write('bad binary')\nsys.exit(1)\n", "bad binary")])
def test_decoder_failure_is_explicit_without_partial_files(tmp_path, monkeypatch, body, reason):
    output = native(tmp_path, "candidate.cubin")
    decoder(tmp_path, monkeypatch, "nvdisasm", body)
    info = instructions.collect(output, "cuda", "ptx", ["disassembly"])["disassembly"]
    assert info["status"] == "failed" and reason in info["reason"]
    assert info["files"] == [] and not (output / "instructions/disassembly").exists()


def test_trainium_retains_chunked_tables_without_inventing_event_order(tmp_path, monkeypatch):
    output = native(tmp_path, "kernel.neff", "kernel.ntff")
    # The transport preserves bytes. These fixtures test the envelope, not Parquet decoding.
    body = """target = Path(sys.argv[sys.argv.index('--output-file') + 1])
target.mkdir()
for name in ['Instruction_0', 'Instruction_1', 'BirInstruction', 'SemaphoreUpdate']:
    (target / (name + '.parquet')).write_bytes(b'PAR1overlapping intervalsPAR1')
(target / 'Other.parquet').write_bytes(b'not requested')
state = Path(sys.argv[sys.argv.index('--data-path') + 1])
state.mkdir()
(state / 'Instruction.parquet').write_bytes(b'PAR1duplicate cached dataPAR1')
"""
    decoder(tmp_path, monkeypatch, "neuron-explorer", body)
    info = instructions.collect(output, "trainium", "nki", ["instruction_timeline"])["instruction_timeline"]
    assert info["status"] == "ok" and info["qualification"] == "unqualified"
    assert info["completeness"] == "not_verified" and info["dropped_events"] is None
    assert info["source_mapping_status"] == "not_checked"
    assert "DmaPacket" in info["missing_tables"]
    paths = [output / item["file"] for item in info["files"]]
    assert not any("state" in path.parts for path in paths)
    assert len([path for path in paths if path.name.startswith("Instruction_")]) == 2
    assert all(path.read_bytes() == b'PAR1overlapping intervalsPAR1' for path in paths if path.suffix == '.parquet')
    assert not any(path.name == 'Other.parquet' for path in paths)


def test_empty_trainium_export_fails(tmp_path, monkeypatch):
    output = native(tmp_path, "kernel.neff", "kernel.ntff")
    decoder(tmp_path, monkeypatch, "neuron-explorer", "print('no tables')\n")
    info = instructions.collect(output, "trainium", "nki", ["instruction_timeline"])["instruction_timeline"]
    assert info["status"] == "failed" and "no Instruction" in info["reason"]


def test_decoder_timeout_and_output_limit(tmp_path):
    for name, code, error in [("timeout", "import time; time.sleep(20)", TimeoutError),
                              ("large", "print('x' * 100000)", ValueError)]:
        directory = tmp_path / name
        directory.mkdir()
        with pytest.raises(error):
            instructions.run_tool([sys.executable, "-c", code], directory, timeout=0.2, max_bytes=1000)


def test_decoder_symlinks_are_rejected(tmp_path):
    output = native(tmp_path, "candidate.cubin")
    directory = tmp_path / "links"
    directory.mkdir()
    (directory / "outside").symlink_to(output / "candidate.cubin")
    with pytest.raises(ValueError, match="symbolic link"):
        instructions.files(directory)


def test_artifact_transport_limit_is_an_outcome(tmp_path, monkeypatch):
    from gpu2tensor import client
    output = native(tmp_path, "candidate.cubin")
    decoder(tmp_path, monkeypatch, "nvdisasm", "print('SASS')\n")
    monkeypatch.setattr(client, "MAX_RESULT_BYTES", 100)
    info = instructions.collect(output, "cuda", "ptx", ["disassembly"])["disassembly"]
    assert info["status"] == "failed" and "result byte limit" in info["reason"]
    assert info["files"] == [] and not (output / "instructions/disassembly").exists()


def test_too_many_files_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(instructions, "MAX_FILES", 2)
    for i in range(3):
        (tmp_path / str(i)).touch()
    with pytest.raises(ValueError, match="file limit"):
        instructions.files(tmp_path)


def test_inspection_failure_does_not_erase_successful_measurement(tmp_path, monkeypatch):
    from gpu2tensor.execute import evaluate
    payload = pack_request(candidate("cpu"), workload(), False, 2, ("disassembly",))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(tmp_path)
    (tmp_path / "output").mkdir()
    def failed_inspection(*args):
        return {"disassembly": {"status": "failed", "reason": "mock decoder error"}}
    monkeypatch.setattr(instructions, "collect", failed_inspection)
    result = evaluate(tmp_path, "cpu")
    assert result["status"] == "ok" and len(result["latency_ms"]) == 2
    assert result["instruction_artifacts"]["disassembly"]["status"] == "failed"


def test_rejected_candidate_skips_inspection(tmp_path, monkeypatch):
    from gpu2tensor import Candidate
    from gpu2tensor.execute import evaluate
    wrong = Candidate("def run(x):\n    return x * 0\n", "python")
    payload = pack_request(wrong, workload(), False, 2, ("disassembly",))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(tmp_path)
    (tmp_path / "output").mkdir()
    def unexpected_inspection(*args):
        pytest.fail("An incorrect candidate must not reach inspection.")
    monkeypatch.setattr(instructions, "collect", unexpected_inspection)
    result = evaluate(tmp_path, "cpu")
    assert result["status"] == "incorrect" and "latency_ms" not in result
