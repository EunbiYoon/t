
# q3.py — Helper to check near-optimality statistics over multiple runs.
from __future__ import annotations
import os, numpy as np
from main import run_experiment
from algorithm import ESConfig
from utils import timestamp

if __name__ == "__main__":
    cfg = ESConfig(P=60, sigma=0.20, alpha=0.10, iters=60, N_eval=15, K=5, seed=2025)
    neurons = (4,3)

    outdir = os.path.join("results", f"q3_{timestamp()}")
    mean, std = run_experiment(
        name="q3_check_near_optimal",
        cfg=cfg,
        neurons=neurons,
        runs=10,
        outdir=outdir,
        save=True,
        progress=True,
    )

    # Report last-iteration stats (mean over runs)
    final_mean = float(mean[-1, 0])
    final_best = float(mean[-1, 1])
    print(f"Final mean J_curr (last iter): {final_mean:.2f}")
    print(f"Final mean J_best (last iter): {final_best:.2f}")
    print("Heuristic near-optimal target is around -120 per the assignment.")
