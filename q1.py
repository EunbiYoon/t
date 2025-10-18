# q1.py
# Part 2 — Q1: Hyperparameter Study (Compute first, plot later)
# Stage 1: run all ES experiments and save .npy
# Stage 2: load all mean/std results and plot them together
# Shows both outer (config) and inner (iteration) progress bars.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe backend for Colima/headless
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithm import ESConfig
from utils import (
    ensure_dir, timestamp, mean_std_over_runs
)
from main import (
    run_experiment
)

# ===============================================================
# User-adjustable constants
# ===============================================================
OUTDIR_BASE = "outputs"
RUNS_PER_CONFIG = 5 #5
ITERS = 30 #30
N_EVAL = 15 #15
K = 5

# ===============================================================
# Configurations (P, sigma, alpha) to test
# ===============================================================
CONFIGS = [
    ("P60_s0.25_a0.50_4", ESConfig(P=60, K=K, sigma=0.5, alpha=0.25,
                                 N_eval=N_EVAL, iters=ITERS))                               
]
NEURONS = [
    (4,)
]
# ===============================================================


def main():
    outdir = os.path.join(OUTDIR_BASE, "q1_" + timestamp())
    ensure_dir(outdir)

    print(f"\n[Q1] Output directory: {outdir}")
    print("[Q1] Stage 1 — Running experiments...\n")

    # =====================
    # Stage 1: Run & Save
    # =====================
    count=0
    for name, cfg in CONFIGS:
        neurons = NEURONS[count]
        print(f"▶ Starting config: {name} (P={cfg.P}, σ={cfg.sigma}, α={cfg.alpha}, neurons={neurons})")

        runs = []
        for hist in tqdm(run_experiment(name, cfg, neurons=neurons,
                                        runs=RUNS_PER_CONFIG,
                                        outdir=outdir, save=True, progress=True)):
            runs.append(hist)
        count += 1


    print("\n✅ Stage 1 complete: all results computed and saved.\n")

    # =====================
    # Stage 2: Plotting
    # =====================
    print("[Q1] Stage 2 — Loading saved results and plotting...\n")

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
    plt.title(f"Q1: Hyperparameter Study — {RUNS_PER_CONFIG} runs each")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = os.path.join(outdir, "q1_study_curves.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("\n✅ [Q1] All done.")
    print(f"===> Final combined plot saved to: {out_png}")
    print(f"===> Directory: {outdir}\n")


if __name__ == "__main__":
    main()
