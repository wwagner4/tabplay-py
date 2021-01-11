import statistics
from dataclasses import dataclass
from typing import List, Any, Tuple

from sklearn.preprocessing import StandardScaler

from localsubm import trainit
from tabplay import Files, Train, MyModel
import matplotlib.pyplot as plt
import numpy as np


def remvals(vals: List[Any], indices: List[int]) -> List[Any]:
    def of_index(v: Any) -> bool:
        i = vals.index(v)
        return i not in indices

    return [v for v in vals if of_index(v)]


@dataclass
class Conf:
    id: int
    scaled: bool
    seed: int
    x_names: List[str]


y_name = 'target'
files = Files()
train = Train()
cnt = 100

train_df = files.train_df()

xscaler = StandardScaler()
yscaler = StandardScaler()


def doit(conf: Conf):
    def f_linreg(x: np.ndarray, y: np.ndarray) -> MyModel:
        return train.fit_linreg(x, y)

    def tr(ni: int) -> Tuple:
        print("processing", conf.x_names[ni])
        xv = remvals(conf.x_names, [ni])
        x = train_df[xv].values
        y = train_df[[train.y_name]].values
        return conf.x_names[ni], [trainit(i + conf.seed + ni, x, y, f_linreg, conf.scaled) for i in range(cnt)]

    x_all = train_df[conf.x_names].values
    y_all = train_df[[train.y_name]].values
    tuple_all = "all", [trainit(conf.seed + i, x_all, y_all, f_linreg, conf.scaled) for i in range(cnt)]
    median_all = statistics.median(tuple_all[1])

    res = [tr(ni) for ni in range(len(conf.x_names))]

    medians = [statistics.median(vals) for _, vals in res]
    min_median = min(medians)
    min_idx = medians.index(min_median)
    print(min_idx)

    res = res + [tuple_all]
    labs = [a for a, b in res]
    vals = [b for a, b in res]

    plt.clf()
    if conf.scaled:
        nam =  f"plt_rm_stepwise_{conf.id}_scaled.png"
        tit = f"Tabular Playground. Stepwise removal {conf.id} scaled"
    else:
        nam =  f"plt_rm_stepwise_{conf.id}.png"
        tit = f"Tabular Playground. Stepwise removal {conf.id}"
    fnam = files.workdir / "plots" / nam
    plt.title(tit)
    plt.ylabel("MSE")
    plt.ylim(0.721, 0.735)
    plt.xticks(rotation=45)
    plt.axhline(median_all, color='b')
    plt.axhline(min_median, color='r')
    plt.boxplot(vals, labels=labs)

    plt.savefig(fnam)
    print(f"Plotted to {fnam.absolute()}")

    if len(conf.x_names) > 2:
        next_conf = Conf(id=conf.id + 1, scaled=conf.scaled, seed=conf.seed + 131, x_names=remvals(conf.x_names, [min_idx]))
        doit(next_conf)


start_conf = Conf(id=700, seed=8979, scaled=True, x_names=train.x_names)
doit(start_conf)
