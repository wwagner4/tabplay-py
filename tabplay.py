import pandas as pd
from pathlib import Path

workdir = Path("/opt") / "work"
datadir = workdir / "data"
train_file = datadir / "train.csv"

train_df = pd.read_csv(str(train_file))
print(train_df.keys())
