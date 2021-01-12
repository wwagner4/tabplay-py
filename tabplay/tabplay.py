import os
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class MyModel(ABC):

    @abstractmethod
    def predict(self, x):
        pass


@dataclass
class GradientBoostingConfig:
    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float

    def __init__(self, learning_rate: float = 0.1,
                 max_depth: int = 3, n_estimators: int = 100,
                 subsample: float = 1.0):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample


class Files:
    workdir: Path
    datadir: Path
    train_file: Path
    test_file: Path

    def __init__(self):
        workdir_str = os.getenv("TABPLAY_WORKDIR", default="/opt/work")
        self.workdir = Path(workdir_str)
        self.datadir = self.workdir / "data"
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

    @staticmethod
    def create_submission(files, predictable, subm_id, test_df, x):
        yp = predictable.predict(x)
        id_df = test_df[['id']]
        target_df = pd.DataFrame(yp, columns=['target'])
        subm_df = id_df.join(target_df)
        subm_file = files.datadir / f"subm_{subm_id}.csv"
        subm_df.to_csv(subm_file, index=False)
        print("wrote to", subm_file.absolute())

    @staticmethod
    def fit_linreg(x: np.ndarray, y: np.ndarray):
        return LinearRegression().fit(x, y)

    @staticmethod
    def fit_gbm(x: np.ndarray, y: np.ndarray, config: GradientBoostingConfig):
        regr = GradientBoostingRegressor(
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            n_estimators=config.n_estimators,
            subsample=config.subsample)
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
