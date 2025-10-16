# utils.py
import os
import numpy as np
from datetime import datetime

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    """Return current timestamp string, e.g. 2025-10-17_02-15-30."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def mean_std_over_runs(runs_list):
    """
    runs_list: list of arrays, each shape (T_i, 2) with columns [J_curr, J_best]
    The T_i may differ; we align to the minimum T to stack safely.
    """
    if len(runs_list) == 0:
        raise ValueError("runs_list is empty")

    # 1) 길이 진단
    lengths = [arr.shape[0] for arr in runs_list]
    min_T = min(lengths)
    if len(set(lengths)) != 1:
        print(f"[warn] Histories have different lengths: {lengths} -> aligning to min_T={min_T}")

    # 2) 최소 길이에 맞춰 자르기
    trimmed = [arr[:min_T] for arr in runs_list]

    # 3) 스택 후 mean/std 계산
    H = np.stack(trimmed, axis=0)  # shape: (R, min_T, 2)
    return H.mean(axis=0), H.std(axis=0)
