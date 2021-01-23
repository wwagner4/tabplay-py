import random as ran

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabplay import Files, Train, MyModel


def main():
    files = Files()
    train = Train()
    scaled = True
    ls_id = '04'

    def f_gbm(x: np.ndarray, y: np.ndarray) -> MyModel:
        c = {'learning_rate': 0.1, 'max_depth': 9}
        return train.fit_gbm(x, y, c)

    def f_rf(x: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_random_forest(x, y, {})

    def f_linreg(x: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_linreg(x, y)

    def f_mean(_: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_mean(y)

    def f_median(_: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_median(y)

    trainall_df = pd.read_csv(str(files.train_file))
    print("traindf.shape", trainall_df.shape)

    ran.seed(123)
    seeds = [ran.randint(0, 100000) for _ in range(15)]

    x_all = trainall_df[train.x_names].values
    y_all = trainall_df[[train.y_name]].values.ravel()
    print("started random forest")
    mse_rf = [train.trainit(se, x_all, y_all, f_rf, scaled) for se in seeds]
    print("finished random forest")
    mse_gbm = [train.trainit(se, x_all, y_all, f_gbm, scaled) for se in seeds]
    print("finished gbm")
    mse_linreg = [train.trainit(se, x_all, y_all, f_linreg, scaled) for se in seeds]
    print("finished lin reg")
    mse_mean = [train.trainit(se, x_all, y_all, f_mean, scaled) for se in seeds]
    print("finished lin mean")
    mse_median = [train.trainit(se, x_all, y_all, f_median, scaled) for se in seeds]
    print("finished lin median")

    if scaled:
        nam = f"plt_compare_models_scaled_{ls_id}.png"
    else:
        nam = f"plt_compare_models_{ls_id}.png"
    fnam = files.workdir / "plots" / nam
    all_data = [mse_rf, mse_gbm, mse_linreg, mse_mean, mse_median]
    all_labels = ["rf", "gbm", "linreg", "mean", "median"]
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
