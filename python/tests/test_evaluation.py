import numpy as np
import pytest

from gpu2tensor import Candidate, Evaluator, Result
from gpu2tensor.data import batches
from gpu2tensor.examples.softmax import candidate, workload


def test_correct_candidate_has_samples_and_can_be_saved(tmp_path):
    result = Evaluator(backend="cpu").evaluate(candidate("cpu"), workload(), repetitions=3)
    assert result.correct
    assert result.record["correctness_cases"] == 4
    assert len(result.record["latency_ms"]) == 3
    result.save(tmp_path)
    assert Result.load(tmp_path).record == result.record
    assert (tmp_path / "candidate.py").exists()
    with pytest.raises(FileExistsError, match="empty"):
        result.save(tmp_path)


def test_wrong_candidate_never_receives_latency():
    wrong = Candidate("def run(x):\n    return x * 0\n", "python")
    result = Evaluator(backend="cpu").evaluate(wrong, workload())
    assert result.record["status"] == "incorrect"
    assert result.latency_ms is None


def test_input_mutation_does_not_change_oracle():
    wrong = Candidate("def run(x):\n    x.zero_()\n    return x\n", "python")
    result = Evaluator(backend="cpu").evaluate(wrong, workload())
    assert not result.correct


def test_syntax_error_is_an_outcome():
    result = Evaluator(backend="cpu").evaluate(Candidate("not valid python!", "python"), workload())
    assert result.record["status"] == "error"
    assert result.record["stage"] == "load"


def test_timeout_returns_without_hanging():
    source = "def run(x):\n    while True:\n        pass\n"
    result = Evaluator(backend="cpu", timeout=3).evaluate(Candidate(source, "python"), workload())
    assert result.record["status"] == "timeout"


def test_batches_are_bounded_mask_missing_and_own_storage():
    consumed = []

    def results():
        for index in range(5):
            consumed.append(index)
            yield Result({"status": "ok", "latency_ms": [index + 1.0]})

    stream = batches(results(), ["latency_ms", "missing.counter"], batch_size=2)
    first = next(stream)
    assert consumed == [0, 1]
    assert first.valid.tolist() == [[True, False], [True, False]]
    tensors = first.tensors()
    assert tensors["values"].data_ptr() == first.values.ctypes.data
    retained = first.values.copy()
    assert len(list(stream)) == 2
    np.testing.assert_array_equal(first.values, retained)


def test_gym_controls_episode_and_preserves_failed_feedback():
    from gpu2tensor.gym import KernelEnv
    env = KernelEnv(Evaluator(backend="cpu"), workload(), language="python",
                    reward=lambda result: float(result.correct), success=lambda result: result.correct)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    observation, reward, terminated, truncated, info = env.step(candidate("cpu").source)
    assert env.observation_space.contains(observation)
    assert reward == 1.0 and terminated and not truncated
    assert info["result"].correct
    with pytest.raises(RuntimeError, match="reset"):
        env.step(candidate("cpu").source)


def test_profile_decoder_excludes_inclusive_scope_totals():
    from gpu2tensor.profiles import proton_kernels
    profile = [{"frame": {"name": "ROOT"}, "metrics": {"time (ns)": 100}, "children": [
        {"frame": {"name": "candidate"}, "metrics": {"time (ns)": 100}, "children": [
            {"frame": {"name": "softmax"},
             "metrics": {"time (ns)": 100, "device_type": "CUDA", "count": 32}}
        ]}
    ]}, {"device": {"clock": 1500}}]
    kernels = proton_kernels(profile)
    assert len(kernels) == 1
    assert kernels[0]["duration_ns"] == 100
    assert kernels[0]["metrics"]["count"] == 32
    with pytest.raises(RuntimeError, match="no CUDA"):
        proton_kernels([])


def test_strided_workload_is_rejected_before_transport():
    from gpu2tensor import Workload
    with pytest.raises(ValueError, match="contiguous"):
        Workload("unused", [(np.zeros((4, 8), dtype=np.float32)[:, ::2],)])


def test_observation_example_does_not_import_gym():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c",
                    "import sys; import gpu2tensor.examples.pretrain; "
                    "assert 'gymnasium' not in sys.modules"], check=True)
