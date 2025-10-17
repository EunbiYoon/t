# main.py
# Shared run helper (run_experiment) + the original PolicyNet demo example.

from __future__ import annotations
import os
import numpy as np

from algorithm import ESConfig, train_es
from utils import ensure_dir, timestamp, mean_std_over_runs


def run_experiment(
    name: str,
    cfg: ESConfig,
    neurons=(3, 2),
    runs: int = 1,
    outdir: str | None = None,
    save: bool = True,
    progress: bool = True,
):
    """
    Run the ES training multiple times and (optionally) save results.

    Parameters
    ----------
    name : str
        A label for this experiment (used as a filename prefix).
    cfg : ESConfig
        Evolution Strategies (ES) hyperparameters.
    neurons : tuple[int, ...]
        Hidden layer sizes for PolicyNet (e.g., (3, 2)).
    runs : int
        Number of independent repetitions to average over.
    outdir : str | None
        Output directory to save results. If None, skip saving.
    save : bool
        If True, save raw runs and mean/std arrays to disk.
    progress : bool
        If True, show a tqdm progress bar over repeated runs.

    Returns
    -------
    runs_list : list[np.ndarray]
        A list of histories, each of shape (T, 2) where columns are [J_curr, J_best].
    mean : np.ndarray
        Mean learning curve over runs, shape (T, 2).
    std : np.ndarray
        Std learning curve over runs, shape (T, 2).
    """
    runs_list = []
    if progress:
        from tqdm import trange
        for _ in trange(runs, desc=f"{name}", dynamic_ncols=True):
            hist, _ = train_es(neurons_per_layer=neurons, cfg=cfg)
            runs_list.append(hist)
    else:
        for _ in range(runs):
            hist, _ = train_es(neurons_per_layer=neurons, cfg=cfg)
            runs_list.append(hist)

    mean, std = mean_std_over_runs(runs_list)

    if save and outdir is not None:
        ensure_dir(outdir)
        np.savez(os.path.join(outdir, f"{name}_runs.npz"), *runs_list)
        np.save(os.path.join(outdir, f"{name}_mean.npy"), mean)
        np.save(os.path.join(outdir, f"{name}_std.npy"), std)

    return runs_list, mean, std


# ---------------------------
# Original PolicyNet demo
# ---------------------------
if __name__ == "__main__":
    import argparse
    from policy import *
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "es-check"], default="es-check")
    args = parser.parse_args()

    if args.mode == "demo":
        # (기존 데모) -----------------------------
        np.set_printoptions(precision=2, suppress=True, linewidth=100, threshold=np.inf)
        neurons_per_layer = (3, 2)
        policy = PolicyNet(neurons_per_layer=neurons_per_layer)
        print(f"\nNetwork architecture: {neurons_per_layer}")
        theta = policy.get_policy_parameters()
        n_parameters = theta.size
        print(f"Total number of parameters: {n_parameters}")
        print(f"\nPolicy parameters:\n{theta}")
        state = np.array([0.9, -0.06])
        print(f"\nState: {state}")
        action = policy.act(state)
        print(f"\nAction selected by the policy: {action}")
        noise = np.random.standard_normal(n_parameters).astype(np.float32)
        print(f"\nNoise that will be added to the policy parameters:\n{noise}")
        new_theta = theta + noise
        policy.set_policy_parameters(new_theta)
        print(f"\nNew policy parameters:\n{policy.get_policy_parameters()}")
        action = policy.act(state)
        print(f"\nAction selected by the updated policy: {action}")

    else:
        # (검증 러너) -----------------------------
        # 추천: 재현성 필요하면 시드 고정
        np.random.seed(0)

        cfg = ESConfig(
            P=60,      # 교수님 제안
            K=5,       # 과제 명시값
            sigma=0.20,
            alpha=0.10,
            N_eval=15,
            iters=30   # 교수님 제안
        )

        name   = "sanity_P60_nn3_only"
        runs   = 5   # 교수님 제안
        neurons= (3,)  # 단일 레이어 3 뉴런

        runs_list, mean, std = run_experiment(
            name=name,
            cfg=cfg,
            neurons=neurons,
            runs=runs,
            outdir="results",
            save=True,
            progress=True
        )

        # 각 run에서 마지막 iteration의 J_curr 확인 (-1000 평평 곡선 여부)
        last_J_curr = [float(hist[-1, 0]) for hist in runs_list]
        print("\nFinal J_curr per run:", last_J_curr)

        # 간단한 판정: 전부 -1000이 아니고, 평균이 -900보다 낫다면 '의미있는' 것으로 간주
        ok_all = all(j > -1000 for j in last_J_curr)
        ok_avg = (np.mean(last_J_curr) > -900)
        if ok_all and ok_avg:
            print("✅ Sanity check PASSED: this setting consistently escapes -1000.")
        else:
            print("❌ Sanity check FAILED: some runs are flat at -1000 or mean too low.")
