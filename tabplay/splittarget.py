from tabplay import Files

files = Files()

df_train = files.train_df()
train_border = 7.94
train_min = 5.0

df_train_left = df_train.loc[df_train['target'] < train_border]
df_train_left = df_train_left.loc[df_train_left['target'] > train_min]
df_train_right = df_train.loc[df_train['target'] >= train_border]

print("all", df_train.shape)
print("left", df_train_left.shape)
print("right", df_train_right.shape)

diff = df_train.shape[0] - df_train_left.shape[0] - df_train_right.shape[0]
print("diff", diff)
