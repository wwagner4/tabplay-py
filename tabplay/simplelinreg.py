import pandas as pd

from tabplay import Files, Train

files = Files()
train = Train()

train_df = pd.read_csv(str(files.train_file))
test_df = pd.read_csv(files.test_file)

x = train_df[train.x_names].values
y = train_df[[train.y_name]].values
esti = train.fit_linreg(x, y)

xt = test_df[train.x_names].values
train.create_submission(files, esti, "simplelinreg_01", test_df, xt)
