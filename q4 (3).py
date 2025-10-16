
# q4.py — Explore stability vs. performance trade-offs for different hyperparameters
from __future__ import annotations
import os
from main import run_experiment
from algorithm import ESConfig
from utils import timestamp

if __name__ == "__main__":
    out_root = os.path.join("results", f"q4_{timestamp()}")
    os.makedirs(out_root, exist_ok=True)

    experiments = [
        ("stable_small_step",   ESConfig(P=60, sigma=0.15, alpha=0.05, iters=60, seed=777)),
        ("fast_but_noisy",      ESConfig(P=40, sigma=0.30, alpha=0.20, iters=60, seed=778)),
        ("compute_heavy_highP", ESConfig(P=120, sigma=0.20, alpha=0.10, iters=60, seed=779)),
        ("low_variance",        ESConfig(P=80, sigma=0.15, alpha=0.10, iters=60, seed=780)),
    ]

    for name, cfg in experiments:
        run_experiment(
            name=f"q4_{name}",
            cfg=cfg,
            neurons=(4,3),
            runs=5,
            outdir=os.path.join(out_root, name),
            save=True,
            progress=True,
        )

    print(f"Saved Q4 analyses under: {out_root}")
