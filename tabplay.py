import pandas as pd
from pathlib import Path

workdir = Path("/opt") / "work"
datadir = workdir / "data"
train_file = datadir / "train.csv"

train_df = pd.read_csv(str(train_file))
x_names = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
           'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']
y_name = ['target']

x = train_df[x_names].values
y = train_df[y_name].values

print(x.shape)
print(y.shape)

