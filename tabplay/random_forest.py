import argparse
import random
from dataclasses import asdict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from localsubm import trainit
from tabplay import Files, Train, MyModel, CvCfg, DataSetCfg, RunCfg

cvs = {
    '01': CvCfg(
        cv_id='random_forest_01',
        title='Random Forest Cross Validation on number of estimators',
        seed=238947,
        ds_cfgs=[
            DataSetCfg(
                ds_id='n_estimators',
                run_cfgs=[
                    RunCfg(
                        run_id="50",
                        cfg={'n_estimators': 50}
                    ),
                    RunCfg(
                        run_id="100",
                        cfg={'n_estimators': 100}
                    ),
                    RunCfg(
                        run_id="200",
                        cfg={'n_estimators': 200}
                    ),
                    RunCfg(
                        run_id="400",
                        cfg={'n_estimators': 400}
                    ),
                ]
            ),
        ]
    ),
    '02': CvCfg(
        cv_id='random_forest_02',
        title='Random Forest CV on number of estimators, NOT scaled',
        seed=2389,
        ds_cfgs=[
            DataSetCfg(
                ds_id='n_estimators',
                run_cfgs=[
                    RunCfg(
                        run_id="50",
                        scaled=False,
                        cfg={'n_estimators': 50}
                    ),
                    RunCfg(
                        run_id="100",
                        scaled=False,
                        cfg={'n_estimators': 100}
                    ),
                    RunCfg(
                        run_id="200",
                        scaled=False,
                        cfg={'n_estimators': 200}
                    ),
                    RunCfg(
                        run_id="400",
                        scaled=False,
                        cfg={'n_estimators': 400}
                    ),
                ]
            ),
        ]
    ),
    '03': CvCfg(
        cv_id='random_forest_03',
        title='Random Forest CV on max_depth, n_estimators',
        seed=2822347,
        ds_cfgs=[
            DataSetCfg(
                ds_id='n_estimators 200',
                run_cfgs=[
                    RunCfg(
                        run_id="5",
                        cfg={'n_estimators': 200, 'max_depth': 5}
                    ),
                    RunCfg(
                        run_id="15",
                        cfg={'n_estimators': 200, 'max_depth': 15}
                    ),
                    RunCfg(
                        run_id="25",
                        cfg={'n_estimators': 200, 'max_depth': 25}
                    ),
                    RunCfg(
                        run_id="auto",
                        cfg={'n_estimators': 200, 'max_depth': None}
                    ),
                ]
            ),
            DataSetCfg(
                ds_id='n_estimators 100',
                run_cfgs=[
                    RunCfg(
                        run_id="5",
                        cfg={'n_estimators': 100, 'max_depth': 5}
                    ),
                    RunCfg(
                        run_id="15",
                        cfg={'n_estimators': 100, 'max_depth': 15}
                    ),
                    RunCfg(
                        run_id="25",
                        cfg={'n_estimators': 100, 'max_depth': 25}
                    ),
                    RunCfg(
                        run_id="auto",
                        cfg={'n_estimators': 100, 'max_depth': None}
                    ),
                ]
            ),
        ]
    ),
}

files = Files()
train = Train()


def run(cv: CvCfg):
    print("cv", cv.cv_id)
    pprint(asdict(cv), width=200)
    random.seed(cv.seed)

    trainall_df = files.train_df().head(10000)
    print("read data", trainall_df.shape)

    x_all = trainall_df[train.x_names].values
    y_all = trainall_df[[train.y_name]].values.ravel()

    def fitit(rc: RunCfg) -> float:
        print("RF fitit", rc.run_id)

        def f_rf(x: np.ndarray, y: np.ndarray) -> MyModel:
            return train.fit_random_forest(x, y, rc.cfg)

        return trainit(random.randint(0, 100000), x_all, y_all, f_rf, rc.scaled)
    for ds_cfg in cv.ds_cfgs:
        print("data set", ds_cfg.ds_id)
        results = [(cfg.run_id, fitit(cfg)) for cfg in ds_cfg.run_cfgs]

        all_data = [r[1] for r in results]
        all_labels = [r[0] for r in results]
        plt.ylim(0.69, 0.75)
        plt.title(cv.title)
        plt.plot(all_labels, all_data)

    legend_vals = [ds.ds_id for ds in cv.ds_cfgs]
    plt.legend(legend_vals)
    plt.axhline(0.699, color='r')
    plt.axhline(0.7013, color='g')

    plot_dir = files.workdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()
    nam = f"cv_{cv.cv_id}.png"
    fnam = plot_dir / nam
    plt.savefig(fnam)
    print(f"Plotted to {fnam.absolute()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("id", choices=cvs.keys(), help="The id to run")
    myargs: argparse.Namespace = parser.parse_args()
    run(cvs[myargs.id])


if __name__ == "__main__":
    main()
