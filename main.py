# main.py
# Shared run helper (run_experiment) + the original PolicyNet demo example.

from __future__ import annotations
import os
import numpy as np

from algorithm import ESConfig, train_es
from utils import ensure_dir, timestamp, mean_std_over_runs

def run_experiment(
    name: str,
    cfg: ESConfig,
    neurons=(3, 2),
    runs: int = 1,
    outdir: str | None = None,
    save: bool = True,
    progress: bool = True,
    save_style: str = "compact",   # "compact" | "full"
):
    """
    Run the ES training multiple times and optionally save results.

    save_style='compact': save only (run.npy, mean.npy, std.npy)
                          run.npy = [J_curr sequence ... , J_best_final]
                          mean/std = single scalar each (final J_curr stats)
    save_style='full'   : save full per-iteration curves (mean.npy (T,2), std.npy (T,2))
    """
    runs_list = []
    if progress:
        from tqdm import trange
        for _ in trange(runs, desc=f"{name}", dynamic_ncols=True):
            hist, _ = train_es(neurons_per_layer=neurons, cfg=cfg)
            runs_list.append(np.asarray(hist))
    else:
        for _ in range(runs):
            hist, _ = train_es(neurons_per_layer=neurons, cfg=cfg)
            runs_list.append(np.asarray(hist))

    # ---- Compute mean/std curves ----
    mean, std = mean_std_over_runs(runs_list)

    # ---- Save results ----
    ensure_dir(outdir)
    # 1️⃣ run.npy : J_curr sequence + final J_best
    np.save(os.path.join(outdir, f"{name}_run.npy"), runs_list)

    # 2️⃣ mean.npy / std.npy : final J_curr scalar stats
    finals = np.array([h[-1, 0] for h in runs_list], dtype=float)
    mean_final = np.array(finals.mean())
    std_final = np.array(finals.std())
    np.save(os.path.join(outdir, f"{name}_mean.npy"), mean_final)
    np.save(os.path.join(outdir, f"{name}_std.npy"), std_final)

    print(f"[run_experiment] Saved compact files:")
    print(f" - {name}_run.npy  shape={runs_list}")
    print(f" - {name}_mean.npy scalar={mean_final}")
    print(f" - {name}_std.npy  scalar={std_final}\n")

    return runs_list, mean, std



# ---------------------------
# Original PolicyNet demo
# ---------------------------
if __name__ == "__main__":
    from policy import *

    np.set_printoptions(precision=2, suppress=True, linewidth=100, threshold=np.inf)

    # Example: build a small policy network with two hidden layers (3 and 2 neurons).
    neurons_per_layer = (3, 2)
    policy = PolicyNet(neurons_per_layer=neurons_per_layer)
    print(f"\nNetwork architecture: {neurons_per_layer}")

    # Fetch the current policy parameters (flattened) and print them.
    theta = policy.get_policy_parameters()
    n_parameters = theta.size
    print(f"Total number of parameters: {n_parameters}")
    print(f"\nPolicy parameters:\n{theta}")

    # Create a sample state [position, velocity].
    state = np.array([0.9, -0.06])
    print(f"\nState: {state}")

    # Query the policy for an action for this state.
    action = policy.act(state)
    print(f"\nAction selected by the policy: {action}")

    # Generate random noise and create a new parameter vector.
    noise = np.random.standard_normal(n_parameters).astype(np.float32)
    print(f"\nNoise that will be added to the policy parameters:\n{noise}")
    new_theta = theta + noise

    # Load the modified parameters and verify that they changed.
    policy.set_policy_parameters(new_theta)
    print(f"\nNew policy parameters:\n{policy.get_policy_parameters()}")

    # Query the updated policy again for the same state.
    action = policy.act(state)
    print(f"\nAction selected by the updated policy: {action}")
