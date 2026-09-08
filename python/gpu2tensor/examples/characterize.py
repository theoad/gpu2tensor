"""Repeat matched profiled/unprofiled evaluations on one CUDA worker."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gpu2tensor import Candidate, Evaluator
from gpu2tensor.examples.softmax import sources, workload


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "cv": float(values.std() / values.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/characterize"))
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Use at least two trials to measure variation.")
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(args.endpoint)
    rowwise, fused = sources()
    guarded = fused.replace("def run(x):", "def run(x):\n    assert float(x[0, 0]) != 123456.0, 'Reused mutated input'")
    guarded = guarded.replace("    return y", "    x.fill_(123456.0)\n    return y")
    report = {"method": "alternating whole-request pairs; SDK import, compile/cache and serialization included",
              "results": {}, "trace_completeness": "not_verified", "dropped_events": None}
    for name, source in (("rowwise", rowwise), ("fused", fused), ("mutation_guard", guarded)):
        program = Candidate(source, "triton", name)
        # Populate caches before comparing whole-request costs.
        warm = evaluator.evaluate(program, workload(), repetitions=5)
        warm.save(args.output / f"{name}-warmup")
        if not warm.correct:
            raise RuntimeError(warm.record)
        trials = []
        for trial in range(args.trials):
            pair = {}
            for profile in ((False, True) if trial % 2 == 0 else (True, False)):
                started = time.perf_counter()
                result = evaluator.evaluate(program, workload(), profile=profile, repetitions=30)
                elapsed = time.perf_counter() - started
                result.save(args.output / f"{name}-{trial}-{int(profile)}")
                if not result.correct or (profile and result.record["profile_status"] != "ok"):
                    raise RuntimeError(result.record)
                if profile and name != "mutation_guard":
                    expected = 32 if name == "rowwise" else 1
                    assert result.record["profile"]["kernels"] == expected, result.record
                pair[str(profile)] = {"request_seconds": elapsed, "latency_ms": result.latency_ms,
                                      "latency_samples_ms": result.record["latency_ms"],
                                      "profile": result.record.get("profile"),
                                      "profile_seconds": result.record.get("profile_seconds")}
                report["host"] = result.record["host"]
                report["device"] = result.record["device"]
                report["ami"] = result.record["ami"]
                print(f"{name}: trial {trial}, profile={profile}, {elapsed:.3f}s", flush=True)
            trials.append(pair)
        report["results"][name] = {
            "trials": trials,
            "unprofiled_latency_ms": summarize([sample for pair in trials for sample in pair["False"]["latency_samples_ms"]]),
            "unprofiled_request_seconds": summarize([pair["False"]["request_seconds"] for pair in trials]),
            "profiled_request_seconds": summarize([pair["True"]["request_seconds"] for pair in trials]),
            "paired_request_delta_seconds": [pair["True"]["request_seconds"] - pair["False"]["request_seconds"] for pair in trials],
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
