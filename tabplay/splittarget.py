import matplotlib.pyplot as plt
import numpy as np

from tabplay import Files, Train, Util


def hist(data: np.ndarray, hist_id: str, title: str):
    plt.clf()

    plt.hist(data, 50, facecolor='green', alpha=0.75)
    plt.ylabel('count')
    plt.xlabel('target')
    plt.title(title)
    plt.axis([5, 10, 0, 10000])
    plt.grid(True)

    plot_dir = files.plotdir
    nam = f"splittarget_{hist_id}.png"
    fnam = plot_dir / nam
    plt.savefig(fnam)
    print("wrote histogran to", fnam.absolute())


files = Files()
train = Train()
util = Util()

df_train = files.train_df()
train_border = 7.94
min_data = 5.0

x_all = df_train[train.x_names].values
y_all = df_train[[train.y_name]].values

_, x, _, y = util.split_arrays_by_value(x_all, y_all, min_data)
xl, xr, yl, yr = util.split_arrays_by_value(x, y, train_border)

print("xl", xl.shape)
print("yl", yl.shape)
print("xr", xr.shape)
print("yr", yr.shape)

hist(yl, 'left', f'target values smaller {train_border:.2f}')
hist(yr, 'right', f'target values greater {train_border:.2f}')
