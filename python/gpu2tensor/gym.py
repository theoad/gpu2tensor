"""Optional Gym wrapper; evaluation itself has no Gym dependency."""

import string

import gymnasium as gym
import numpy as np

from gpu2tensor import Candidate


class KernelEnv(gym.Env):
    """Each action submits complete source; reward and success are client callbacks."""

    metadata = {"render_modes": []}

    def __init__(self, evaluator, workload, *, language, reward, success=None,
                 max_steps=8, profile=False, repetitions=30, artifacts=()):
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self.evaluator = evaluator
        self.workload = workload
        self.language = language
        self.reward = reward
        self.success = success or (lambda result: False)
        self.max_steps = max_steps
        self.profile = profile
        self.repetitions = repetitions
        from gpu2tensor.instructions import requested
        self.artifacts = requested(artifacts, profile)
        self.action_space = gym.spaces.Text(max_length=65536, min_length=1, charset=string.printable)
        self.observation_space = gym.spaces.Dict({
            "correct": gym.spaces.Discrete(2),
            "measured": gym.spaces.Discrete(2),
            "latency_ms": gym.spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
        })
        self.steps = 0
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.steps = 0
        self.done = False
        return {"correct": 0, "measured": 0, "latency_ms": np.zeros(1, dtype=np.float32)}, {}

    def step(self, source):
        if self.done:
            raise RuntimeError("Call reset() before starting another episode.")
        if not self.action_space.contains(source):
            raise ValueError("Submit a nonempty ASCII Python source string, up to 65536 characters.")
        options = {"profile": self.profile, "repetitions": self.repetitions}
        if self.artifacts:
            options["artifacts"] = self.artifacts
        result = self.evaluator.evaluate(Candidate(source, self.language), self.workload, **options)
        self.steps += 1
        terminated = bool(self.success(result))
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        observation = {"correct": int(result.correct), "measured": int(result.latency_ms is not None),
                       "latency_ms": np.array([result.latency_ms or 0.0], dtype=np.float32)}
        return observation, float(self.reward(result)), terminated, truncated, {"result": result}
