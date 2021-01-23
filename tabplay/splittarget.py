import matplotlib.pyplot as plt
import numpy as np

from tabplay import Files, Train, Util, MyModel

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


def straight_forward(x: np.ndarray, y: np.ndarray) -> MyModel:
    class M(MyModel):
        train_border: float = 7.94
        model_left: MyModel
        model_right: MyModel

        def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
            xl, xr, yl, yr = util.split_arrays_by_value(x_data, y_data, self.train_border)
            self.model_left = train.fit_gbm(xl, yl, train.gbm_optimal_config)
            self.model_right = train.fit_gbm(xr, yr, train.gbm_optimal_config)

        def predict(self, x_test: np.ndarray) -> np.ndarray:
            pl = self.model_left.predict(x_test)
            pr = self.model_right.predict(x_test)
            print("pl", pl.shape)
            print("pr", pr.shape)
            return np.maximum(pl, pr)

    return M(x, y)


def no_split(x: np.ndarray, y: np.ndarray) -> MyModel:
    class M(MyModel):
        model: MyModel

        def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
            self.model = train.fit_gbm(x_data, y_data, train.gbm_optimal_config)

        def predict(self, x_test: np.ndarray) -> np.ndarray:
            return self.model.predict(x_test)

    return M(x, y)


def main():
    min_data = 5.0

    df_train = files.train_df()
    x_all = df_train[train.x_names].values
    y_all = df_train[[train.y_name]].values
    _, x, _, y = util.split_arrays_by_value(x_all, y_all, min_data)
    print('x', x.shape)
    print('y', y.shape)

    e1 = train.trainit(82374294, x, y, no_split, False)
    e2 = train.trainit(12380, x, y, straight_forward, False)

    print("no split ", e1)
    print("staight_forward ", e2)


if __name__ == '__main__':
    main()
