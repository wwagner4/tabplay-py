import random as ran

import matplotlib.pyplot as plt
import numpy as np

from localsubm import trainit
from tabplay import Files, Train, MyModel


def main():
    files = Files()
    train = Train()
    scaled = True
    ls_id = '05'

    def f_gbm(x: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_gbm(x, y)

    trainall_df = files.train_df()
    print("traindf.shape", trainall_df.shape)

    ran.seed(123)
    shuffles = [ran.randint(0, 100000) for _ in range(15)]

    x = trainall_df[train.x_names].values
    y = trainall_df[[train.y_name]].values.ravel()
    mse_gbm = [trainit(_shuffle, x, y, f_gbm, scaled) for _shuffle in shuffles]
    print("finished gbm")

    if scaled:
        nam = f"gbm_cv_{ls_id}_scaled.png"
        tit = "GBM Cross Validation scaled"
    else:
        nam = f"gbm_cv_{ls_id}.png"
        tit = "GBM Cross Validation"
    fnam = files.workdir / "plots" / nam
    all_data = [mse_gbm]
    all_labels = ["gbm"]
    plt.ylim(0.69, 0.75)
    plt.title(tit)
    plt.axhline(0.699, color='r')
    plt.axhline(0.7013, color='g')
    plt.axhline(0.7278, linestyle='--')
    plt.axhline(0.7349, linestyle='--')
    plt.boxplot(all_data, labels=all_labels)

    plt.savefig(fnam)
    print(f"Plottet to {fnam.absolute()}")


if __name__ == "__main__":
    main()
