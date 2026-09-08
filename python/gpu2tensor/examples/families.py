"""Classify execution layout across held-out workload families, with controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpu2tensor import Candidate, Evaluator, Result
from gpu2tensor.data import batches
from gpu2tensor.examples import affine, softmax
from gpu2tensor.examples.pretrain import fit, normalize_training_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint")
    source.add_argument("--reuse", action="store_true")
    parser.add_argument("--backend", choices=["cuda", "trainium"], default="cuda")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/families"))
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("Use at least one permutation control.")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(args.endpoint) if args.endpoint else None
    records, labels, families = [], [], []
    for family, module in enumerate((softmax, affine)):
        for seed in range(3 * family, 3 * family + 3):
            for rows in (8, 16, 32, 64):
                sources = module.sources() if args.backend == "cuda" else module.trainium_sources()
                language = "triton" if args.backend == "cuda" else "nki"
                programs = [Candidate(source, language, f"{module.__name__}-{label}")
                            for label, source in enumerate(sources)]
                for label, program in enumerate(programs):
                    path = args.output / f"family-{family}-seed-{seed}-rows-{rows}-class-{label}"
                    if args.reuse:
                        result = Result.load(path)
                    else:
                        result = evaluator.evaluate(program, module.workload(rows=rows, seed=seed), profile=True)
                        result.save(path)
                    if not result.correct or result.record["profile_status"] != "ok":
                        raise RuntimeError(result.record)
                    if args.backend == "cuda":
                        assert result.record["profile"]["kernels"] == (rows if label == 0 else 1)
                    result.artifacts.clear()
                    records.append(result)
                    labels.append(label)
                    families.append(family)
                print(f"family={family}, seed={seed}, rows={rows}", flush=True)
    columns = (["profile.kernels", "profile.device_time_ns"] if args.backend == "cuda" else
               ["profile.measurements.scalar_engine_instruction_count", "profile.measurements.vector_engine_instruction_count"])
    tensors = next(batches(records, columns, batch_size=len(records))).tensors(args.device)
    assert bool(tensors["valid"].all().cpu())
    raw = torch.log1p(tensors["values"])
    target = torch.tensor(labels, dtype=torch.int64, device=args.device)
    report = {"task": "rowwise_or_fused_execution_layout", "target_device": records[0].record["device"],
              "backend": args.backend,
              "target_host": records[0].record["host"], "ami": records[0].record["ami"],
              "learner_device": args.device, "columns": columns, "folds": []}
    for held_out in (0, 1):
        test = torch.tensor(np.array(families) == held_out, device=args.device)
        train = ~test
        features, mean, scale, active = normalize_training_features(raw, train)
        model, measured = fit(features, target, train, test, args.device)
        _, zeros = fit(torch.zeros_like(features), target, train, test, args.device)
        _, count_only = fit(features[:, :1], target, train, test, args.device)
        _, time_only = fit(features[:, 1:], target, train, test, args.device)
        controls = []
        for seed in range(args.permutations):
            shuffled = target.clone()
            indices = np.random.default_rng(seed + 100).permutation(int(train.sum().cpu()))
            shuffled[train] = target[train][torch.tensor(indices, device=args.device)]
            # Only training labels are permuted; test labels remain fixed.
            _, control = fit(features, shuffled, train, test, args.device)
            controls.append(control)
        p = (1 + sum(item["test_cross_entropy"] <= measured["test_cross_entropy"] for item in controls)) / (1 + len(controls))
        report["folds"].append({"held_out_family": ["softmax", "affine"][held_out],
                                "active_features": active.cpu().tolist(),
                                "train_samples": int(train.sum().cpu()), "test_samples": int(test.sum().cpu()),
                                "measured": measured, "zero_features": zeros,
                                "first_feature_only": count_only, "second_feature_only": time_only,
                                "permuted_training_labels": controls, "permutation_p_cross_entropy": p})
        torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                    "mean": mean.cpu(), "scale": scale.cpu(), "active": active.cpu()}, args.output / f"held-out-{held_out}.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
