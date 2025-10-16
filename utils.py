
# utils.py
from __future__ import annotations
import os, time
import numpy as np
from typing import List,Tuple
def ensure_dir(path: str):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def mean_std_over_runs(histories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std across repeated runs.

    histories: list of arrays shaped (T, 2); T may differ across runs.
    Returns (mean, std) each shape (T, 2) for a common trimmed T.
    """
    if not histories:
        raise ValueError("mean_std_over_runs: empty histories")

    # 1) 방어: 2D shape 보장
    for i, h in enumerate(histories):
        if not isinstance(h, np.ndarray) or h.ndim != 2 or h.shape[1] != 2:
            raise ValueError(f"histories[{i}] has invalid shape {getattr(h, 'shape', None)}; expected (T,2)")

    # 2) 공통 길이로 잘라서 스택
    min_T = min(h.shape[0] for h in histories)
    if min_T == 0:
        raise ValueError("mean_std_over_runs: at least one history has length 0")
    H = np.stack([h[:min_T, :] for h in histories], axis=0)  # (R, min_T, 2)

    return H.mean(axis=0), H.std(axis=0)

