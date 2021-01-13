import argparse
import random as ran
from dataclasses import dataclass
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from localsubm import trainit
from tabplay import Files, Train, MyModel, RandomForestConfig


@dataclass
class RfRunCfg:
    rid: str
    seed: int
    scaled: bool
    cfg: RandomForestConfig


@dataclass
class RfCv:
    run_id: str
    title: str
    cnt: int
    runCfgs: List[RfRunCfg]


cvs = {
    "01": RfCv(
        run_id="01",
        title="Random Forest Cross Validation on number of estimators",
        cnt=15,
        runCfgs=[
            RfRunCfg(
                rid="50",
                seed=3847,
                scaled=True,
                cfg=RandomForestConfig(n_estimators=50)
            ),
            RfRunCfg(
                rid="100",
                seed=9237,
                scaled=True,
                cfg=RandomForestConfig(n_estimators=100)
            ),
            RfRunCfg(
                rid="200",
                seed=92222847,
                scaled=True,
                cfg=RandomForestConfig(n_estimators=200)
            ),
            RfRunCfg(
                rid="400",
                seed=92452847,
                scaled=True,
                cfg=RandomForestConfig(n_estimators=400)
            ),
        ]
    ),
    "02": RfCv(
        run_id="02",
        title="Random Forest Cross Validation on number of estimators, NOT scaled",
        cnt=15,
        runCfgs=[
            RfRunCfg(
                rid="50",
                seed=3847,
                scaled=False,
                cfg=RandomForestConfig(n_estimators=50)
            ),
            RfRunCfg(
                rid="100",
                seed=9237,
                scaled=False,
                cfg=RandomForestConfig(n_estimators=100)
            ),
            RfRunCfg(
                rid="200",
                seed=92222847,
                scaled=False,
                cfg=RandomForestConfig(n_estimators=200)
            ),
            RfRunCfg(
                rid="400",
                seed=92452847,
                scaled=False,
                cfg=RandomForestConfig(n_estimators=400)
            ),
        ]
    ),
}

files = Files()
train = Train()


def run(cv: RfCv):
    print("cv on rf")
    pprint(cv)

    trainall_df = files.train_df()
    print("read data", trainall_df.shape)

    x_all = trainall_df[train.x_names].values
    y_all = trainall_df[[train.y_name]].values.ravel()

    def fitit(rc: RfRunCfg) -> List[float]:
        print("RF fitit", rc.rid)

        def f_rf(x: np.ndarray, y: np.ndarray) -> MyModel:
            return train.fit_random_forest(x, y, rc.cfg)

        ran.seed(rc.seed)
        seeds = [ran.randint(0, 100000) for _ in range(cv.cnt)]

        return [trainit(s, x_all, y_all, f_rf, rc.scaled) for s in seeds]

    results = [(cfg.rid, fitit(cfg)) for cfg in cv.runCfgs]

    nam = f"rf_cv_{cv.run_id}_scaled.png"
    plot_dir = files.workdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()
    fnam = plot_dir / nam
    all_data = [r[1] for r in results]
    all_labels = [r[0] for r in results]
    plt.ylim(0.69, 0.75)
    plt.title(cv.title)
    plt.axhline(0.699, color='r')
    plt.axhline(0.7013, color='g')
    plt.boxplot(all_data, labels=all_labels)

    plt.savefig(fnam)
    print(f"Plotted to {fnam.absolute()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("id", choices=cvs.keys(), help="The id to run")
    myargs: argparse.Namespace = parser.parse_args()
    run(cvs[myargs.id])


if __name__ == "__main__":
    main()
