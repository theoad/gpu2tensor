from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from gpu2tensor import Candidate, Pool, Result


def test_pool_completion_order_backpressure_and_ownership(monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    active = set()
    lock = threading.Lock()

    class Worker:
        def __init__(self, endpoint, timeout):
            self.endpoint = endpoint
            self.index = int(endpoint[-1])

        def evaluate(self, candidate, workload, **kwargs):
            with lock:
                assert self.index not in active
                active.add(self.index)
            entered[self.index].set()
            assert release[self.index].wait(5)
            with lock:
                active.remove(self.index)
            return Result({"status": "ok", "name": candidate.name})

    monkeypatch.setattr("gpu2tensor.pool.Evaluator", Worker)
    consumed = []

    def source():
        for index in range(5):
            consumed.append(index)
            yield Candidate("unused", "python", str(index))

    pool = Pool(["http://worker0", "http://worker1"])
    stream = pool.observe(source(), None)
    with ThreadPoolExecutor(max_workers=1) as caller:
        first = caller.submit(next, stream)
        try:
            assert all(event.wait(5) for event in entered)
            assert consumed == [0, 1]
            release[1].set()
            result = first.result(timeout=5)
            assert result.record["name"] == "1"
            assert result.record["worker_endpoint"] == "http://worker1"
            assert consumed == [0, 1]
            with pytest.raises(RuntimeError, match="existing iterator"):
                next(pool.observe([], None))
        finally:
            for event in release:
                event.set()
        remaining = list(stream)
    assert sorted(item.record["name"] for item in remaining) == ["0", "2", "3", "4"]
    assert not active and not pool.active


def test_pool_closing_and_errors_release_ownership(monkeypatch):
    class Worker:
        def __init__(self, endpoint, timeout):
            self.endpoint = endpoint

        def evaluate(self, candidate, workload, **kwargs):
            if candidate.name == "bad":
                raise OSError("Worker disconnected")
            return Result({"status": "ok"})

    monkeypatch.setattr("gpu2tensor.pool.Evaluator", Worker)
    pool = Pool(["http://worker"])
    stream = pool.observe([Candidate("", "", "ok")] * 10, None)
    next(stream)
    stream.close()
    assert not pool.active
    with pytest.raises(OSError, match="disconnected"):
        list(pool.observe([Candidate("", "", "bad")], None))
    assert not pool.active
    assert list(pool.observe([], None)) == []


def test_pool_rejects_duplicate_endpoints():
    with pytest.raises(ValueError, match="duplicate"):
        Pool(["http://worker", "http://worker/"])
