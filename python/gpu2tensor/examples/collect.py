"""Compare serial and bounded parallel collection on isolated worker devices."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gpu2tensor import Candidate, Evaluator, Pool
from gpu2tensor.data import batches
from gpu2tensor.examples import affine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--backend", choices=["cuda", "trainium"], required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/collect"))
    args = parser.parse_args()
    if len(args.endpoints) < 2 or args.jobs < len(args.endpoints) or args.trials < 1:
        parser.error("Use two or more endpoints, at least one job per worker, and a positive trial count.")
    args.output.mkdir(parents=True, exist_ok=True)
    program, target = affine.candidate(args.backend), affine.workload()
    ownership = []
    for index, endpoint in enumerate(args.endpoints):
        result = Evaluator(endpoint).evaluate(program, target, profile=True, repetitions=5)
        result.save(args.output / f"worker-{index}-warmup")
        if not result.correct or result.record["profile_status"] != "ok":
            raise RuntimeError(result.record)
        ownership.append({key: result.record.get(key) for key in ("host", "device", "neuron_core", "ami", "backend_sha256")})
    report = {"backend": args.backend, "workers": ownership, "trials": [],
              "method": "alternating serial/pool trials after per-worker warmup; one profiled job per device at a time"}
    for trial in range(args.trials):
        pair = {}
        for mode in (("serial", "pool") if trial % 2 == 0 else ("pool", "serial")):
            candidates = [Candidate(program.source, program.language, f"job-{index}") for index in range(args.jobs)]
            pool = Pool(args.endpoints[:1] if mode == "serial" else args.endpoints)
            results = []
            started = time.perf_counter()
            for result in pool.observe(candidates, target, profile=True, repetitions=5):
                result.save(args.output / f"trial-{trial}-{mode}-{result.record['name']}")
                if not result.correct or result.record["profile_status"] != "ok":
                    raise RuntimeError(result.record)
                result.artifacts.clear()
                results.append(result)
            seconds = time.perf_counter() - started
            assert {result.record["name"] for result in results} == {program.name for program in candidates}
            batch = next(batches(results, ["latency_ms"], batch_size=args.jobs)).tensors()
            assert bool(batch["valid"].all())
            pair[mode] = {"seconds": seconds, "jobs_per_second": args.jobs / seconds,
                          "worker_endpoints": [r.record["worker_endpoint"] for r in results],
                          "prepare_and_check_seconds": [r.record["prepare_and_check_seconds"] for r in results],
                          "profile_seconds": [r.record["profile_seconds"] for r in results],
                          "kernel_latency_ms": [r.latency_ms for r in results]}
            print(json.dumps({"trial": trial, "mode": mode, **pair[mode]}), flush=True)
        pair["speedup"] = pair["serial"]["seconds"] / pair["pool"]["seconds"]
        report["trials"].append(pair)
    report["median_speedup"] = float(np.median([pair["speedup"] for pair in report["trials"]]))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
