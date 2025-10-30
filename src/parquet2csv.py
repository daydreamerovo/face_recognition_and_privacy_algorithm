import pandas as pd

path = '../data/landmarks_dataset.parquet'
output_path_csv = "../data/landmarks_dataset.csv"
df = pd.read_parquet(path)
csv = df.to_csv(output_path_csv, index=False)