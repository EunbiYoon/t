#!/usr/bin/env python3
# plot_from_csv5_fixed.py
# 지정된 CSV 5개를 label과 함께 그리기 (명령줄 입력 필요 없음)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# ✏️ 여기서 파일 경로랑 라벨 수정하면 됨
# =====================================================
csv_files = [
    "q1-2-1/detail_P60_s0.25_a0.50_2.csv",
    "q1-2-2/detail_P60_s0.25_a0.50_4.csv",
    "q1-2-3/detail_P60_s0.25_a0.50_22.csv",
    "q1-2-4/detail_P60_s0.25_a0.50_44.csv",
    "q1-2-5/detail_P60_s0.25_a0.50_23.csv",
    "q1-2-6/detail_P60_s0.25_a0.50_32.csv",
]
labels = [
    "NN (2)",
    "NN (4)",
    "NN (2,2)",
    "NN (4,4)",
    "NN (2,3)",
    "NN (3,2)"
]
title = "Q1: Neural Network Structure Comparison (P=60, σ=0.25, α=0.5)"
outdir = Path("outputs/trick2")
outdir.mkdir(parents=True, exist_ok=True)
# =====================================================

plt.figure(figsize=(9, 7))

for path, lab in zip(csv_files, labels):
    df = pd.read_csv("outputs/"+path)
    if not {"Iteration", "mean", "std"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: Iteration, mean, std")
    it = df["Iteration"].to_numpy()
    mu = df["mean"].to_numpy(dtype=float)
    sd = df["std"].to_numpy(dtype=float)
    order = np.argsort(it)
    it, mu, sd = it[order], mu[order], sd[order]
    plt.plot(it, mu, label=lab)
    plt.fill_between(it, mu - sd, mu + sd, alpha=0.15)

plt.xlabel("ES Iteration")
plt.ylabel("Return (higher is better)")
plt.title(title)
plt.legend(fontsize=9, ncol=2, loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_png = outdir / "curves_csv5.png"
plt.savefig(out_png, dpi=150)
