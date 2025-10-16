
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
    Compute mean and std learning curves over multiple ES runs.

    Parameters
    ----------
    histories : list of np.ndarray
        Each element is a (T,2) array, columns = [J_curr, J_best].

    Returns
    -------
    mean : np.ndarray
        Mean over runs, shape (T,2)
    std : np.ndarray
        Std over runs, shape (T,2)
    """
    if not histories:
        raise ValueError("histories is empty (no runs provided)")

    # Validate and normalize
    valid = []
    for i, h in enumerate(histories):
        arr = np.asarray(h)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"histories[{i}] has invalid shape {getattr(arr, 'shape', None)}; expected (T,2)"
            )
        valid.append(arr[:, :2])

    # Handle unequal lengths (trim to minimum)
    lengths = [h.shape[0] for h in valid]
    min_T = min(lengths)
    if len(set(lengths)) != 1:
        print(f"[mean_std_over_runs] Warning: unequal run lengths {lengths}, trimming to {min_T}")
    valid = [h[:min_T, :] for h in valid]

    H = np.stack(valid, axis=0)   # (R,T,2)
    mean = H.mean(axis=0)         # (T,2)
    std = H.std(axis=0)           # (T,2)
    return mean, std

