import numpy as np
import os
import pandas as pd

# === 1️⃣ 측정 폴더 경로만 지정 ===
# 예: results_dir = "results"  또는 절대경로
results_dir = "outputs/q1_2025-10-17_17-42-29"

# === 2️⃣ 자동 탐색 ===
records = []
for file in os.listdir(results_dir):
    if file.endswith("_mean.npy"):
        prefix = file.replace("_mean.npy", "")
        mean_path = os.path.join(results_dir, f"{prefix}_mean.npy")
        std_path  = os.path.join(results_dir, f"{prefix}_std.npy")
        if not os.path.exists(std_path):
            continue  # 짝이 없으면 skip

        mean = np.load(mean_path)
        std  = np.load(std_path)

        # 마지막 iteration의 J_curr(mean[:,0], std[:,0])
        final_mean = float(mean[-1,0])
        final_std  = float(std[-1,0])

        # prefix parsing (예: P60_s0.25_a0.5 → 60, 0.25, 0.5)
        parts = prefix.split("_")
        p = parts[0].replace("P", "")
        s = parts[1].replace("s", "")
        a = parts[2].replace("a", "")

        records.append({
            "prefix": prefix,
            "P": int(float(p)),
            "sigma": float(s),
            "alpha": float(a),
            "mean": round(final_mean, 2),
            "std": round(final_std, 2)
        })

# === 3️⃣ 결과를 CSV로 저장 ===
df = pd.DataFrame(records).sort_values(["P","sigma","alpha"])
out_csv = os.path.join(results_dir, "summary_results.csv")
df.to_csv(out_csv, index=False)

print(f"✅ Saved summary to: {out_csv}")
print(df)
