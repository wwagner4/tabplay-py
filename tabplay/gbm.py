import argparse
import random as ran
from dataclasses import dataclass
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from localsubm import trainit
from tabplay import Files, Train, MyModel, GradientBoostingConfig


@dataclass
class GbmRunCfg:
    rid: str
    seed: int
    scaled: bool
    cfg: GradientBoostingConfig


@dataclass
class GbmCv:
    run_id: str
    title: str
    cnt: int
    runCfgs: List[GbmRunCfg]


cvs = {
    "01": GbmCv(
        run_id="01",
        title="GBM Cross Validation",
        cnt=15,
        runCfgs=[
            GbmRunCfg(
                rid="3",
                seed=3847,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=3)
            ),
            GbmRunCfg(
                rid="5",
                seed=9237,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=5)
            ),
            GbmRunCfg(
                rid="9",
                seed=92847,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=9)
            ),
        ]
    ),
    "02": GbmCv(
        run_id="02",
        title="GBM CV on max depth, learning_rate=0.1",
        cnt=15,
        runCfgs=[
            GbmRunCfg(
                rid="5",
                seed=9237,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=5)
            ),
            GbmRunCfg(
                rid="8",
                seed=9237,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=8)
            ),
            GbmRunCfg(
                rid="9",
                seed=92847,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=9)
            ),
            GbmRunCfg(
                rid="10",
                seed=383347,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=10)
            ),
            GbmRunCfg(
                rid="12",
                seed=924537,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=12)
            ),
            GbmRunCfg(
                rid="15",
                seed=9265847,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=15)
            ),
        ]
    ),
    "03": GbmCv(
        run_id="03",
        title="GBM CV on learning rate, max depth = 9",
        cnt=15,
        runCfgs=[
            GbmRunCfg(
                rid="0.05",
                seed=383347,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.05, max_depth=9)
            ),
            GbmRunCfg(
                rid="0.1",
                seed=383347,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.1, max_depth=9)
            ),
            GbmRunCfg(
                rid="0.15",
                seed=383347,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.15, max_depth=9)
            ),
            GbmRunCfg(
                rid="0.2",
                seed=924537,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.2, max_depth=9)
            ),
            GbmRunCfg(
                rid="0.15",
                seed=9265847,
                scaled=True,
                cfg=GradientBoostingConfig(learning_rate=0.25, max_depth=9)
            ),
        ]
    ),
}

files = Files()
train = Train()


def run(cv: GbmCv):
    print("cv on gbm")
    pprint(cv)

    trainall_df = files.train_df()
    print("read data", trainall_df.shape)

    x_all = trainall_df[train.x_names].values
    y_all = trainall_df[[train.y_name]].values.ravel()

    def fitit(rc: GbmRunCfg) -> List[float]:
        print("fitit", rc.rid)

        def f_gbm(x: np.ndarray, y: np.ndarray) -> MyModel:
            return train.fit_gbm(x, y, rc.cfg)

        ran.seed(rc.seed)
        seeds = [ran.randint(0, 100000) for _ in range(cv.cnt)]

        return [trainit(s, x_all, y_all, f_gbm, rc.scaled) for s in seeds]

    results = [(cfg.rid, fitit(cfg)) for cfg in cv.runCfgs]

    nam = f"gbm_cv_{cv.run_id}_scaled.png"
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
