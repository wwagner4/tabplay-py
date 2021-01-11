import random as ran
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabplay import Files, Train, MyModel


def trainit(seed: int, x: np.ndarray, y: np.ndarray,
            f: Callable[[np.ndarray, np.ndarray], MyModel], scale: bool) -> float:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
    xscaler = StandardScaler()
    if scale:
        xscaler.fit(x_train)
        x_train = xscaler.transform(x_train, copy=True)
        x_test = xscaler.transform(x_test, copy=True)
    esti = f(x_train, y_train)
    yp = esti.predict(x_test)
    return mean_squared_error(y_test, yp, squared=False)


def main():
    files = Files()
    train = Train()
    scaled = True
    ls_id = '04'

    def f_linreg(x: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_linreg(x, y)

    def f_gbm(x: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_gbm(x, y)

    def f_mean(_: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_mean(y)

    def f_median(_: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_median(y)

    trainall_df = pd.read_csv(str(files.train_file))
    print("traindf.shape", trainall_df.shape)

    ran.seed(123)
    shuffles = [ran.randint(0, 100000) for _ in range(15)]

    x = trainall_df[train.x_names].values
    y = trainall_df[[train.y_name]].values.ravel()
    mse_gbm = [trainit(_shuffle, x, y, f_gbm, scaled) for _shuffle in shuffles]
    print("finished gbm")
    mse_linreg = [trainit(_shuffle, x, y, f_linreg, scaled) for _shuffle in shuffles]
    print("finished lin reg")
    mse_mean = [trainit(_shuffle, x, y, f_mean, scaled) for _shuffle in shuffles]
    print("finished lin mean")
    mse_median = [trainit(_shuffle, x, y, f_median, scaled) for _shuffle in shuffles]
    print("finished lin median")


    if scaled:
        nam = f"plt_compare_models_scaled_{ls_id}.png"
    else:
        nam = f"plt_compare_models_{ls_id}.png"
    fnam = files.workdir / "plots" / nam
    all_data = [mse_gbm, mse_linreg, mse_mean, mse_median]
    all_labels = ["gbm", "linreg", "mean", "median"]
    plt.ylim(0.69, 0.75)
    plt.title("Tabular Playground with submissions")
    plt.axhline(0.699, color='r')
    plt.axhline(0.7013, color='g')
    plt.axhline(0.7278, linestyle='--')
    plt.axhline(0.7349, linestyle='--')
    plt.boxplot(all_data, labels=all_labels)

    plt.savefig(fnam)
    print(f"Plottet to {fnam.absolute()}")


if __name__ == "__main__":
    main()
