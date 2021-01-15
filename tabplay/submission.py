from dataclasses import dataclass, asdict
from pprint import pprint
from typing import Callable

import pandas as pd
import numpy as np

from tabplay import Files, Train, MyModel, Util, GradientBoostingConfig

files = Files()
train = Train()
util = Util()


@dataclass
class SubmConfig:
    s_id: str
    s_description: str
    f_model: Callable[[np.ndarray, np.ndarray], MyModel]


configs = {
    '01': SubmConfig(
        s_id='linreg_01',
        s_description='Linear regression on all predictors',
        f_model=train.fit_linreg
    ),
    '02': SubmConfig(
        s_id='gbm_02',
        s_description='Predict target with a gradient boost regressor. max depth: 9, learning_rate: 0.1',
        f_model=lambda x, y: train.fit_gbm(x, y,
                                           GradientBoostingConfig(learning_rate=0.1, max_depth=9))
    ),

}


def run(cfg: SubmConfig):
    print("create submission", cfg.s_id)
    pprint(asdict(cfg), width=200)
    train_df = pd.read_csv(str(files.train_file))
    test_df = pd.read_csv(files.test_file)

    x = train_df[train.x_names].values
    y = train_df[[train.y_name]].values.ravel()
    print("---> train model")
    esti = cfg.f_model(x, y)
    print("<--- train model")

    xt = test_df[train.x_names].values
    subm_file = files.datadir / f"subm_{cfg.s_id}.csv"

    print("---> predict test")
    train.create_submission(subm_file, esti, test_df, xt)
    print("<--- predict test")
    print("finished submission", cfg.s_id)
    print("finished submission", cfg.s_description)
    print("finished submission", subm_file.absolute())


def main():
    cfg = util.parse_config_by_id(configs)
    run(cfg)


if __name__ == '__main__':
    main()
