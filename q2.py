# q2.py
# Part 2 — Q2: Run 15 times with best hyperparameter

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe backend for Colima/headless
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithm import ESConfig
from utils import (
    ensure_dir, timestamp
)
from main import (
    run_experiment
)

# ===============================================================
# User-adjustable constants
# ===============================================================
OUTDIR_BASE = "outputs"
NEURONS = (3, 2)
RUNS_PER_CONFIG = 15 #5
ITERS = 30 #40
N_EVAL = 15 #15
K = 5

# ===============================================================
# Configurations (P, sigma, alpha) to test
# ===============================================================
CONFIGS = [
    ("P60_s0.25_a0.5", ESConfig(P=60, K=K, sigma=0.25, alpha=0.5,
                                 N_eval=N_EVAL, iters=ITERS)),
]
# ===============================================================


def main():
    outdir = os.path.join(OUTDIR_BASE, "q2_" + timestamp())
    ensure_dir(outdir)

    print(f"\n[Q2] Output directory: {outdir}")
    print("[Q2] Stage 2 — Running experiments...\n")

    # =====================
    # Stage 1: Run & Save
    # =====================
    for name, cfg in CONFIGS:
        print(f"▶ Starting config: {name} (P={cfg.P}, σ={cfg.sigma}, α={cfg.alpha})")
        runs = []

        # tqdm: outer progress (5 runs)
        for hist in tqdm(run_experiment(name, cfg, neurons=NEURONS, runs=RUNS_PER_CONFIG,outdir=outdir, save=True, progress=True)):
            runs.append(hist)

    print("\n✅ Stage 1 complete: all results computed and saved.\n")

    # =====================
    # Stage 2: Plotting
    # =====================
    print("[Q2] Stage 2 — Loading saved results and plotting...\n")

    plt.figure(figsize=(8, 6))
    for name, _ in CONFIGS:
        mean_path = os.path.join(outdir, f"{name}_mean.npy")
        std_path = os.path.join(outdir, f"{name}_std.npy")

        if not os.path.exists(mean_path):
            print(f"⚠️ Skipping {name}: mean.npy not found")
            continue

        mean = np.load(mean_path)
        std = np.load(std_path)
        x = np.arange(1, mean.shape[0] + 1)
        plt.plot(x, mean[:, 0], label=name)
        plt.fill_between(x, mean[:, 0] - std[:, 0],
                         mean[:, 0] + std[:, 0], alpha=0.15)
        print(f"✅ Added curve for {name}")

    plt.xlabel("ES Iteration")
    plt.ylabel("Return (higher is better)")
    plt.title(f"Q2: Best configuration — {RUNS_PER_CONFIG} runs each")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = os.path.join(outdir, "q2_best_curves.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("\n✅ [Q2] All done.")
    print(f"===> Final combined plot saved to: {out_png}")
    print(f"===> Directory: {outdir}\n")


if __name__ == "__main__":
    main()
