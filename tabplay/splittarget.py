import matplotlib.pyplot as plt
import numpy as np

from tabplay import Files, Train, Util

files = Files()
train = Train()
util = Util()


def all_hists(x_data: np.ndarray, y_data: np.ndarray):
    def hist(data: np.ndarray, hist_id: str, title: str, color: str):
        plt.clf()

        plt.hist(x=data, bins=100, facecolor=color, alpha=0.75)
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

    train_border = 7.94
    xl, xr, yl, yr = util.split_arrays_by_value(x_data, y_data, train_border)
    hist(y_data, 'all', f'target values', color='r')
    hist(yl, 'left', f'target values smaller {train_border:.2f}', color='g')
    hist(yr, 'right', f'target values greater {train_border:.2f}', color='b')


def main():
    min_data = 5.0

    df_train = files.train_df()
    x_all = df_train[train.x_names].values
    y_all = df_train[[train.y_name]].values
    _, x, _, y = util.split_arrays_by_value(x_all, y_all, min_data)
    print('x', x.shape)
    print('y', y.shape)

    all_hists(x, y)


if __name__ == '__main__':
    main()