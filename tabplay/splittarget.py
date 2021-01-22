from tabplay import Files, Train, Util

files = Files()
train = Train()
util = Util()

df_train = files.train_df().head(20000)
train_border = 7.94
train_min = 5.0

x = df_train[train.x_names].values
y = df_train[[train.y_name]].values

xl, xr, yl, yr = util.split_arrays_by_value(x, y, train_border)

print("xl", xl.shape)
print("yl", yl.shape)
print("xr", xr.shape)
print("yr", yr.shape)
