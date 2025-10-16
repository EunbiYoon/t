
# q2.py — Programming Q2 experiments (15 runs using a chosen good config)
from __future__ import annotations
import os
from main import run_experiment
from algorithm import ESConfig
from utils import timestamp

if __name__ == "__main__":
    # Fill in with the best hyperparameters you found from Q1.
    # These defaults are reasonable starting points.
    best_cfg = ESConfig(P=60, sigma=0.20, alpha=0.10, iters=60, N_eval=15, K=5, seed=123)
    neurons = (4, 3)

    outdir = os.path.join("results", f"q2_{timestamp()}")
    res = run_experiment(
        name="q2_best_mean±std_over_15",
        cfg=best_cfg,
        neurons=neurons,
        runs=15,              # Q2: average over 15 runs
        outdir=outdir,
        save=True,
        progress=True,
    )
    print(f"Saved Q2 results under: {outdir}")
