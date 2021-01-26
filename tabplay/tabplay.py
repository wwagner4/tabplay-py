import argparse
import os
import sys
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RunCfg:
    run_id: str
    cfg: dict
    scaled: bool = True


@dataclass
class DataSetCfg:
    ds_id: str
    run_cfgs: List[RunCfg]


@dataclass
class CvCfg:
    cv_id: str
    title: str
    ds_cfgs: List[DataSetCfg]
    seed: int = 1827391


class MyModel(ABC):

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class Files:
    workdir: Path
    datadir: Path
    plotdir: Path
    train_file: Path
    test_file: Path

    def __init__(self):
        workdir_str = os.getenv("TABPLAY_WORKDIR", default="/opt/work")
        self.workdir = Path(workdir_str)
        self.datadir = self.workdir / "data"
        self.plotdir = self.workdir / "plots"
        self.train_file = self.datadir / "train.csv"
        self.test_file = self.datadir / "test.csv"

    def train_df(self) -> pd.DataFrame:
        return pd.read_csv(self.train_file)

    def test_df(self) -> pd.DataFrame:
        return pd.read_csv(self.train_file)


class Train:
    x_names = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
               'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13',
               'cont14']
    y_name = 'target'

    gbm_optimal_config = {'learning_rate': 0.1, 'max_depth': 9}

    @staticmethod
    def trainit(seed: int, x: np.ndarray, y: np.ndarray,
                f: Callable[[np.ndarray, np.ndarray], MyModel],
                scale: bool) -> float:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
        xscaler = StandardScaler()
        if scale:
            xscaler.fit(x_train)
            x_train = xscaler.transform(x_train, copy=True)
            x_test = xscaler.transform(x_test, copy=True)
        my_model = f(x_train, y_train)
        yp = my_model.predict(x_test)
        return mean_squared_error(y_test, yp, squared=False)

    @staticmethod
    def create_submission(subm_file: Path, predictable: Any, test_df, x):
        yp = predictable.predict(x)
        id_df = test_df[['id']]
        target_df = pd.DataFrame(yp, columns=['target'])
        subm_df = id_df.join(target_df)
        subm_df.to_csv(subm_file, index=False)

    @staticmethod
    def fit_linreg(x: np.ndarray, y: np.ndarray) -> MyModel:
        return LinearRegression().fit(x, y)

    @staticmethod
    def fit_gbm(x: np.ndarray, y: np.ndarray, config: dict) -> Any:
        regr = GradientBoostingRegressor(**config)
        resu = regr.fit(x, y.ravel())
        return resu

    @staticmethod
    def fit_random_forest(x: np.ndarray, y: np.ndarray, config: dict) -> Any:
        regr = RandomForestRegressor(**config)
        return regr.fit(x, y)

    @staticmethod
    def fit_mean(y: np.ndarray) -> MyModel:
        _mean = y.mean()

        class M(MyModel):

            def predict(self, x):
                return np.full((x.shape[0], 1), _mean)

        return M()

    @staticmethod
    def fit_median(y: np.ndarray) -> MyModel:
        _mean = np.median(y)

        class M(MyModel):

            def predict(self, x):
                return np.full((x.shape[0], 1), _mean)

        return M()


class Util:

    @staticmethod
    def parse_config_by_id(configs: dict) -> Any:
        parser = argparse.ArgumentParser()
        parser.add_argument("id", choices=configs.keys(), help="The id to run")
        myargs: argparse.Namespace = parser.parse_args()
        return configs[myargs.id]

    @staticmethod
    def split_arrays_by_value(x: np.ndarray, y: np.ndarray, split_value: float):
        y_idx = y.flatten() < split_value
        x1 = x[y_idx, :]
        if len(y.shape) == 1:
            y1 = y[y_idx]
        else:
            y1 = y[y_idx, :]
        y_idx = np.invert(y_idx)
        x2 = x[y_idx, :]
        if len(y.shape) == 1:
            y2 = y[y_idx]
        else:
            y2 = y[y_idx, :]
        return x1, x2, y1, y2

    @staticmethod
    def mean_of_greatest(*vectors: np.ndarray, trigger: float = 0.9) -> np.ndarray:
        def mog(row: np.ndarray) -> float:
            _max = row.max(initial=sys.float_info.min)
            return np.array([x for x in row if x >= _max * trigger]).mean()

        return np.array([mog(row) for row in np.array([*vectors]).T])

    @staticmethod
    def cut_middle(xd: np.ndarray, yd: np.ndarray, min_val: float, max_val: float) -> (np.ndarray, np.ndarray):
        xl1, _, yl1, _ = Util.split_arrays_by_value(xd, yd, max_val)
        _, xo, _, yo = Util.split_arrays_by_value(xl1, yl1, min_val)
        return xo, yo
