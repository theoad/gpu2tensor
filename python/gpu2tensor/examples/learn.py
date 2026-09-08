"""Small MPS/CPU policy experiment with complete source actions and real feedback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpu2tensor import Candidate, Evaluator
from gpu2tensor.data import batches
from gpu2tensor.examples.softmax import sources, workload
from gpu2tensor.gym import KernelEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("artifacts/learn"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    target = workload()
    evaluator = Evaluator(args.endpoint)
    programs = sources()

    # First use observation-only collection, independent of the policy and Gym.
    observed = []
    for index, result in enumerate(evaluator.observe(
            [Candidate(source, "triton", name) for source, name in zip(programs, ["rowwise", "fused"])], target)):
        result.save(args.output / f"observation-{index}")
        if not result.correct or result.record["profile_status"] != "ok":
            raise RuntimeError(f"Observation failed: {result.record}")
        observed.append(result)
    batch = next(batches(iter(observed), ["latency_ms", "profile.kernels"], batch_size=2))
    tensors = batch.tensors(args.device)
    baseline_ms = observed[0].latency_ms

    # This is a deliberately tiny two-program bandit, not compiler synthesis.
    env = KernelEnv(evaluator, target, language="triton", max_steps=args.steps,
                    reward=lambda result: 0.0 if not result.correct else
                    float(np.clip(1.0 - result.latency_ms / baseline_ms, -1.0, 1.0)))
    logits = torch.nn.Parameter(torch.zeros(2, device=args.device))
    optimizer = torch.optim.Adam([logits], lr=0.15)
    env.reset(seed=7)
    history = []
    for step in range(args.steps):
        distribution = torch.distributions.Categorical(logits=logits)
        # Sample on CPU for reproducibility across learner devices.
        action = int(torch.multinomial(distribution.probs.detach().cpu(), 1))
        observation, reward, terminated, truncated, info = env.step(programs[action])
        info["result"].save(args.output / f"step-{step:03d}")
        loss = -distribution.log_prob(torch.tensor(action, device=args.device)) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"step": step, "action": action, "reward": reward,
                        "latency_ms": info["result"].latency_ms, "correct": info["result"].correct})
        print(json.dumps(history[-1]), flush=True)
        if terminated or truncated:
            break
    probabilities = logits.softmax(0).detach().cpu().tolist()
    report = {"experiment": "two_supplied_program_bandit", "learner_device": args.device,
              "target_device": observed[0].record["device"], "target_host": observed[0].record["host"],
              "initial_probabilities": [0.5, 0.5], "final_probabilities": probabilities,
              "observed_latency_ms": [result.latency_ms for result in observed],
              "batch_shape": list(tensors["values"].shape), "steps": history}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"logits": logits.detach().cpu(), "report": report}, args.output / "policy.pt")
    print(json.dumps({key: value for key, value in report.items() if key != "steps"}), flush=True)


if __name__ == "__main__":
    main()
