# algorithm.py
# Evolution Strategies (ES) loop that reuses starter PolicyNet API.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from tqdm import tqdm

from dynamics import MountainCarEnv
from policy import PolicyNet  # starter network (PyTorch) with act/get/set API

# ===============================================================
# Config Dataclass
# ===============================================================
@dataclass
class ESConfig:
    P: int = 50          # population size
    K: int = 5           # top-k elites
    sigma: float = 0.20  # exploration std
    alpha: float = 0.10  # step size
    N_eval: int = 15     # episodes per policy evaluation
    iters: int = 50      # ES iterations


# ===============================================================
# Helper functions
# ===============================================================
def run_episode(env, policy, gamma=1.0, seed=None):
    G = 0.0
    t = 0
    s = env.reset()
    done = False
    while not done:
        a = policy.act(s)
        s, r, done, _ = env.step(a)  # tuple 언패킹
        G += (gamma ** t) * r
        t += 1
    return G




def estimate_J(policy: PolicyNet, N: int = 15) -> float:
    """Estimate expected return J(θ) by averaging N evaluation episodes."""
    returns: List[float] = []
    for _ in range(N):
        env = MountainCarEnv()
        returns.append(run_episode(env, policy, gamma=1.0))
    return float(np.mean(returns)) if returns else -np.inf


# ===============================================================
# One ES Step  — global RNG 사용
# ===============================================================
def es_step(policy: PolicyNet, cfg: ESConfig) -> Tuple[float, float]:
    """
    One ES update:
      - Evaluate current J_curr
      - Sample P perturbations eps ~ N(0, I)
      - Evaluate J_i for each perturbed theta_i
      - Sort by performance **descending** (top-K) and update
        grad ≈ (1/(σK)) * Σ_k (J_k - J_curr) * eps_k  (then L2-normalize)
    Returns: (J_curr, J_best_perturbed)
    """
    theta = policy.get_policy_parameters()
    n = theta.size

    J_curr = estimate_J(policy, N=cfg.N_eval)

    # ✅ 전역 RNG (main.py에서 np.random.seed으로 고정됨)
    eps = np.random.standard_normal(size=(cfg.P, n)).astype(np.float32)

    cand: List[Tuple[float, np.ndarray]] = []
    for i in range(cfg.P):
        theta_i = theta + cfg.sigma * eps[i]
        policy.set_policy_parameters(theta_i)
        J_i = estimate_J(policy, N=cfg.N_eval)
        cand.append((float(J_i), eps[i].copy()))

    # Sort descending
    cand.sort(key=lambda x: x[0], reverse=True)
    top = cand[:cfg.K]
    J_best = top[0][0]
    print(J_best)

    # Gradient estimate and update
    grad = np.zeros_like(theta, dtype=np.float32)
    for (J_k, eps_k) in top:
        grad += (J_k - J_curr) * eps_k
    grad *= (1.0 / (cfg.sigma * cfg.K))

    # Normalize for stability
    norm = float(np.linalg.norm(grad) + 1e-8)
    grad /= norm

    policy.set_policy_parameters(theta + cfg.alpha * grad)
    return J_curr, J_best


# ===============================================================
# Main ES Training Loop
# ===============================================================
def train_es(neurons_per_layer=(3, 2),
             cfg: ESConfig = ESConfig()) -> Tuple[np.ndarray, PolicyNet]:
    """
    Full ES training loop reusing starter PolicyNet.
    Returns:
        history: np.ndarray of shape (iters, 2) with [J_curr, J_best_perturbed] per iter
        policy:  trained PolicyNet
    """
    policy = PolicyNet(neurons_per_layer=neurons_per_layer)
    hist = np.zeros((cfg.iters, 2), dtype=np.float32)

    print(f"    → Starting ES run ({cfg.iters} iterations, P={cfg.P})")

    with tqdm(total=cfg.iters, desc="      inner progress", dynamic_ncols=True, leave=False) as bar:
        for t in range(cfg.iters):
            J_curr, J_best = es_step(policy, cfg)
            hist[t, 0] = J_curr
            hist[t, 1] = J_best
            bar.set_postfix({"J_curr": f"{J_curr:7.2f}", "J_best": f"{J_best:7.2f}"})
            bar.update(1)
    return hist, policy
