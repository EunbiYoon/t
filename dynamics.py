
# dynamics.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

# Reuse the exact constants from policy.py to keep everything in sync
from policy import MIN_POSITION, MAX_POSITION, MIN_VELOCITY, MAX_VELOCITY, TIMEOUT

@dataclass
class MountainCarEnv:
    max_steps: int = TIMEOUT
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.state = None
        self.steps = 0

    def reset(self):
        # S0 = (X0, 0) with X0 ~ Uniform[-0.6, -0.4]
        x0 = self.rng.uniform(-0.6, -0.4)
        self.state = np.array([x0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, a: float):
        """
        Action space: {-1, 0, +1} (Reverse, Neutral, Forward)
        Deterministic dynamics per assignment spec.
        Returns: next_state, reward, done, info
        """
        x, v = float(self.state[0]), float(self.state[1])
        a = float(a)

        v = v + 0.001 * a - 0.0025 * math.cos(3.0 * x)
        v = float(np.clip(v, MIN_VELOCITY, MAX_VELOCITY))
        x = x + v
        x = float(np.clip(x, MIN_POSITION, MAX_POSITION))
        if x == MIN_POSITION or x == MAX_POSITION:
            v = 0.0

        self.state[:] = (x, v)
        self.steps += 1

        done = (x == MAX_POSITION) or (self.steps >= self.max_steps)
        reward = 0.0 if (x == MAX_POSITION) else -1.0
        return self.state.copy(), reward, done, {}

    @property
    def observation(self):
        return None if self.state is None else self.state.copy()
