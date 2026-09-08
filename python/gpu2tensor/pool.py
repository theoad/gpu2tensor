"""Bounded observation collection across operator-owned worker endpoints."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Lock

from gpu2tensor.client import Evaluator


class Pool:
    """Keep at most one evaluation active or waiting to be consumed per endpoint.

    Each endpoint must own a separate device (or an isolated device partition).
    Results arrive in completion order and retain their worker endpoint.
    Closing an iterator waits for its already-running evaluations to finish.
    """

    def __init__(self, endpoints, *, timeout=300):
        if isinstance(endpoints, str):
            raise TypeError("Pass a list of worker endpoints.")
        self.endpoints = tuple(endpoint.rstrip("/") for endpoint in endpoints)
        if not self.endpoints or len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("Provide at least one worker, without duplicate endpoints.")
        self.workers = tuple(Evaluator(endpoint, timeout=timeout) for endpoint in self.endpoints)
        self._lease = Lock()

    @property
    def active(self):
        return self._lease.locked()

    def observe(self, candidates, workload, *, profile=True, repetitions=30, artifacts=()):
        """Yield results with backpressure; never enqueue the whole corpus."""
        from gpu2tensor.instructions import requested
        artifacts = requested(artifacts, profile)
        if not self._lease.acquire(blocking=False):
            raise RuntimeError("Finish or close this pool's existing iterator first.")
        try:
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                pending = {}
                source = iter(candidates)

                def submit(worker):
                    try:
                        candidate = next(source)
                    except StopIteration:
                        return
                    options = {"profile": profile, "repetitions": repetitions}
                    if artifacts:
                        options["artifacts"] = artifacts
                    future = executor.submit(worker.evaluate, candidate, workload, **options)
                    pending[future] = worker

                for worker in self.workers:
                    submit(worker)
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        worker = pending.pop(future)
                        result = future.result()
                        result.record["worker_endpoint"] = worker.endpoint
                        yield result
                        # Refill only when the consumer asks for another result.
                        submit(worker)
        finally:
            self._lease.release()
