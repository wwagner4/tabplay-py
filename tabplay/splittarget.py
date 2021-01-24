import multiprocessing
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from tabplay import Files, Train, Util, MyModel

train_border: float = 7.94
min_data = 5.0
files = Files()


class SplitModels:
    @staticmethod
    def tuple_model(x: np.ndarray, y: np.ndarray) -> MyModel:
        class M(MyModel):
            model_left: MyModel
            model_right: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data, train_border)
                self.model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
                self.model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                pl = self.model_left.predict(x_test)
                pr = self.model_right.predict(x_test)
                print("pl", pl.shape)
                print("pr", pr.shape)
                return np.maximum(pl, pr)

        return M(x, y)

    @staticmethod
    def triple_model(x: np.ndarray, y: np.ndarray) -> MyModel:
        class M(MyModel):
            model_all: MyModel
            model_left: MyModel
            model_right: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data, train_border)
                self.model_all = Train.fit_gbm(x_data, y_data, Train.gbm_optimal_config)
                self.model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
                self.model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                pa = self.model_all.predict(x_test)
                pl = self.model_left.predict(x_test)
                pr = self.model_right.predict(x_test)
                print("pa", pa.shape)
                print("pl", pl.shape)
                print("pr", pr.shape)
                return np.maximum(pa, pl, pr)

        return M(x, y)

    @staticmethod
    def no_split(x: np.ndarray, y: np.ndarray) -> MyModel:
        class M(MyModel):
            model: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                self.model = Train.fit_gbm(x_data, y_data, Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                return self.model.predict(x_test)

        return M(x, y)


def hist(data: np.ndarray, hist_id: str, title: str, color: str):
    plt.clf()
    plt.hist(x=data, bins=100, facecolor=color, alpha=0.75)
    plt.ylabel('count')
    plt.xlabel('target')
    plt.title(title)
    plt.axis([5, 10, 0, 10000])
    plt.grid(True)
    plot_dir = Files.plotdir
    nam = f"splittarget_{hist_id}.png"
    fnam = plot_dir / nam
    plt.savefig(fnam)
    print("wrote histogran to", fnam.absolute())


def all_hists(x_data: np.ndarray, y_data: np.ndarray):
    xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data, train_border)
    hist(y_data, 'all', f'target values', color='r')
    hist(yl, 'left', f'target values smaller {train_border:.2f}', color='g')
    hist(yr, 'right', f'target values greater {train_border:.2f}', color='b')


def hist_predictions(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=203842039)
    xl, xr, yl, yr = Util.split_arrays_by_value(x_train, y_train, train_border)
    model_all = Train.fit_gbm(x_train, y_train, Train.gbm_optimal_config)
    model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
    model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

    yp_all = model_all.predict(x_test)
    yp_left = model_left.predict(x_test)
    yp_right = model_right.predict(x_test)

    hist(yp_all, 'pred_all', f'predicted values', color='orange')
    hist(yp_left, 'pred_left', f'predicted values smaller {train_border:.2f}', color='orange')
    hist(yp_right, 'pred_right', f'predicted values greater {train_border:.2f}', color='orange')


@dataclass
class SplitTrain:
    desc: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    f: Callable


def run_split_train(split_train: SplitTrain) -> (str, float):
    return split_train.desc, Train.trainit(split_train.seed, split_train.x, split_train.y, split_train.f, False)


def train_it(x, y):
    split_trains = [
        SplitTrain("no split", 1213, x, y, SplitModels.no_split),
        SplitTrain("no split", 1323, x, y, SplitModels.no_split),
        SplitTrain("no split", 1223, x, y, SplitModels.no_split),
        SplitTrain("no split", 1233, x, y, SplitModels.no_split),
        SplitTrain("no split", 1232, x, y, SplitModels.no_split),
        SplitTrain("tuple", 83823, x, y, SplitModels.tuple_model),
        SplitTrain("triple", 19283, x, y, SplitModels.triple_model),
        SplitTrain("triple", 1983, x, y, SplitModels.triple_model),
        SplitTrain("triple", 195283, x, y, SplitModels.triple_model),
        SplitTrain("triple", 192683, x, y, SplitModels.triple_model),
    ]

    with multiprocessing.Pool() as pool:
        for desc, error in pool.map(run_split_train, split_trains):
            print(desc, error)


def main():
    df_train = files.train_df()
    x_all = df_train[Train.x_names].values
    y_all = df_train[[Train.y_name]].values
    _, x, _, y = Util.split_arrays_by_value(x_all, y_all, min_data)
    print('x', x.shape)
    print('y', y.shape)

    train_it(x, y)


if __name__ == '__main__':
    main()
