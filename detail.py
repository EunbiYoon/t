import numpy as np
import pandas as pd
import os

# --------------------------------------------------
# 🔧 폴더 경로 지정
# --------------------------------------------------
FOLDER = "outputs/q1-2-6"
RUN_FILE = "P60_s0.25_a0.50_runs.npz"
OUTPUT_FILE="[detail]"+RUN_FILE+".csv"

data = np.load(os.path.join(FOLDER, RUN_FILE))

# --------------------------------------------------
# NPZ 구조 표준화 (3D 형태로 변환)
# --------------------------------------------------
if len(data.files) == 1 and data[data.files[0]].ndim == 3:
    runs_3d = data[data.files[0]]  # (num_runs, num_iters, 2)
else:
    keys = sorted(data.files, key=lambda k: (k.isdigit(), k))
    arrays = [data[k] for k in keys]
    assert all(a.ndim == 2 and a.shape[1] >= 2 for a in arrays), "Unexpected array shape in NPZ"
    runs_3d = np.stack(arrays, axis=0)

num_runs, num_iters, num_cols = runs_3d.shape
print(f"Loaded runs: {num_runs} runs, {num_iters} iterations, {num_cols} cols (J_curr, J_best)")

# --------------------------------------------------
# J_curr (column 0) 기준
# --------------------------------------------------
J = runs_3d[:, :, 0]  # shape = (num_runs, num_iters)

# --------------------------------------------------
# iteration별 mean, std 계산
# --------------------------------------------------
mean = J.mean(axis=0)
std = J.std(axis=0)

# --------------------------------------------------
# DataFrame 생성 (mean, std를 오른쪽에 붙이기)
# --------------------------------------------------
df = pd.DataFrame(
    J.T,
    columns=[f"Run_{i+1}" for i in range(num_runs)],
    index=np.arange(1, num_iters + 1)
)
df.index.name = "Iteration"

# mean/std 추가
df["mean"] = mean
df["std"] = std

# --------------------------------------------------
# 정수 반올림
# --------------------------------------------------
df = df.round(0).astype(int)

# --------------------------------------------------
# CSV 저장
# --------------------------------------------------
output_csv = os.path.join(FOLDER, OUTPUT_FILE)
df.to_csv(output_csv)
print(f"✅ Saved: {output_csv}")
