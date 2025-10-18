import pandas as pd

# CSV 파일 불러오기 (탭 구분으로 되어 있다면 sep='\t')
df = pd.read_csv("outputs/q2/runs_detail.csv", sep='\t')

# 각 행을 LaTeX 형태로 변환
for _, row in df.iterrows():
    values = " & ".join(str(v) for v in row.values)
    values.replace(",","&")
    print(values + " \\\\")
