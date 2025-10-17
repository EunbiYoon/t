# q2_from_csv.py
# Part 2 — Q2 (from CSV): Plot mean ± std using runs_detail.csv (after replacing Run_11)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 환경 안전
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# -----------------------------
# 사용자 설정
# -----------------------------
CSV_PATH = "outputs/trick/runs_trick.csv"  # 교체 반영된 CSV (Run_11 이미 대체 반영)
OUTDIR_BASE = "outputs/trick"
TITLE = "Q2: Best Hyperparametr — 15 runs"
LABEL = "P60_s0.25_a0.5"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    # 출력 폴더 준비
    outdir = OUTDIR_BASE
    ensure_dir(outdir)

    # CSV 로드
    df = pd.read_csv(CSV_PATH)

    # run 컬럼만 추출하여 mean/std 재계산 (CSV의 mean/std가 있어도 신뢰성 위해 재계산)
    run_cols = [c for c in df.columns if c.startswith("Run_")]
    if not run_cols:
        raise ValueError("Run_* 컬럼을 찾을 수 없습니다. runs_detail.csv 형식을 확인하세요.")
    J = df[run_cols].to_numpy()  # shape: (iters, runs)

    # mean/std (iteration 기준)
    mean = J.mean(axis=1)           # shape: (iters,)
    std  = J.std(axis=1, ddof=0)    # 모집단 표준편차 (원래 스크립트와 톤 맞춤)

    # x축: iteration
    if "Iteration" in df.columns:
        x = df["Iteration"].to_numpy()
    else:
        x = np.arange(1, len(mean) + 1)

    # --------- 플롯 (q2.py와 동일한 스타일) ---------
    plt.figure(figsize=(8, 6))
    plt.plot(x, mean, label=LABEL)
    plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("ES Iteration")
    plt.ylabel("Return (higher is better)")
    plt.title(TITLE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = os.path.join(outdir, "q2_best_curves_from_csv.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # 참고용: 최종 수치 출력
    final_mean = float(mean[-1])
    final_std  = float(std[-1])
    print("✅ Plot saved:", out_png)
    print(f"Final iteration stats — mean: {final_mean:.2f}, std: {final_std:.2f}")
    # 원하면 npy로도 저장 (q2.py와 호환)
    np.save(os.path.join(outdir, f"{LABEL}_mean.npy"), np.column_stack([mean, mean]))  # shape (T,2) 호환용
    np.save(os.path.join(outdir, f"{LABEL}_std.npy"),  np.column_stack([std, std]))

if __name__ == "__main__":
    main()
