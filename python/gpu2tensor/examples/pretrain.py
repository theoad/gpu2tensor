"""Observation-only profile classification, with held-out inputs and a control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpu2tensor import Candidate, Evaluator, Result
from gpu2tensor.data import batches
from gpu2tensor.examples.softmax import sources, workload


def normalize_training_features(features, train):
    """Fit scale on training data and ignore counters it cannot teach us about."""
    mean = features[train].mean(0)
    spread = features[train].std(0)
    active = spread > 1e-6
    scale = torch.where(active, spread, torch.ones_like(spread))
    normalized = torch.where(active, (features - mean) / scale, torch.zeros_like(features))
    return normalized, mean, scale, active


def fit(features, labels, train, test, device):
    torch.manual_seed(19)
    model = torch.nn.Linear(features.shape[1], 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = float(torch.nn.functional.cross_entropy(model(features[train]), labels[train]).detach().cpu())
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(model(features[train]), labels[train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = model(features[test]).argmax(1)
        accuracy = float((predictions == labels[test]).float().mean().cpu())
        test_loss = float(torch.nn.functional.cross_entropy(model(features[test]), labels[test]).cpu())
    return model, {"initial_loss": initial, "final_loss": float(loss.detach().cpu()),
                   "test_accuracy": accuracy, "test_cross_entropy": test_loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--endpoint")
    inputs.add_argument("--reuse", action="store_true", help="Train again from saved measurements without executing targets")
    inputs.add_argument("--dataset", type=Path, help="Train from a JSON file of recorded measurements")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pretrain"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(args.endpoint) if args.endpoint else None
    saved = {}
    if args.dataset:
        dataset = json.loads(args.dataset.read_text())
        if dataset["version"] != 1:
            raise ValueError("Unsupported example dataset version.")
        for sample in dataset["samples"]:
            key = (sample["seed"], sample["rows"], sample["label"])
            if key in saved:
                raise ValueError("Duplicate sample in example dataset.")
            saved[key] = sample["record"]
        expected = {(seed, rows, label) for seed in range(6) for rows in (8, 16, 32, 64) for label in range(2)}
        if saved.keys() != expected:
            raise ValueError("Expected both program classes at all 24 input settings.")
    programs = [Candidate(source, "triton", name) for source, name in zip(sources(), ["rowwise", "fused"])]
    records, labels, seeds = [], [], []
    for seed in range(6):
        for rows in (8, 16, 32, 64):
            if args.dataset:
                results = (Result(saved[(seed, rows, label)]) for label in range(2))
            elif args.reuse:
                results = (Result.load(args.output / f"seed-{seed}-rows-{rows}-program-{label}") for label in range(2))
            else:
                results = evaluator.observe(programs, workload(rows=rows, seed=seed))
            for label, result in enumerate(results):
                if not args.reuse:
                    result.save(args.output / f"seed-{seed}-rows-{rows}-program-{label}")
                if not result.correct or result.record["profile_status"] != "ok":
                    raise RuntimeError(f"Observation failed: {result.record}")
                # Raw artifacts are already on disk; retain only small metadata.
                result.artifacts.clear()
                records.append(result)
                labels.append(label)
                seeds.append(seed)
            print(f"{'Loaded' if args.reuse or args.dataset else 'Collected'} seed={seed}, rows={rows}", flush=True)

    columns = ("profile.kernels", "profile.device_time_ns")
    tensors = [batch.tensors(args.device) for batch in batches(iter(records), columns, batch_size=8)]
    if not all(bool(item["valid"].all().cpu()) for item in tensors):
        raise RuntimeError("Required profile measurements are missing.")
    features = torch.log1p(torch.cat([item["values"] for item in tensors]))
    target = torch.tensor(labels, dtype=torch.int64, device=args.device)
    train = torch.tensor(np.array(seeds) < 4, device=args.device)
    test = ~train
    # Fit normalization on training observations only.
    features, mean, scale, active = normalize_training_features(features, train)
    model, measured = fit(features, target, train, test, args.device)
    shuffled = target.clone()
    rng = np.random.default_rng(31)
    shuffled[train] = target[train][torch.tensor(rng.permutation(int(train.sum().cpu())), device=args.device)]
    _, control = fit(features, shuffled, train, test, args.device)
    _, no_measurements = fit(torch.zeros_like(features), target, train, test, args.device)
    report = {"experiment": "profile_classification_smoke_check", "learner_device": args.device,
              "target_device": records[0].record["device"], "target_host": records[0].record["host"],
              "columns": columns, "train_samples": int(train.sum().cpu()), "test_samples": int(test.sum().cpu()),
              "split": "seeds 0-3 train; seeds 4-5 test; both program classes at every shape",
              "measured": measured, "shuffled_training_labels": control,
              "no_measurements": no_measurements, "majority_accuracy": 0.5}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "mean": mean.cpu(), "scale": scale.cpu(), "active": active.cpu(), "report": report}, args.output / "classifier.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
