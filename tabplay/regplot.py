import math
from dataclasses import dataclass
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tabplay import Files


@dataclass
class RegConf:
    nam_x: str
    nam_y: str


files = Files()


def contour_all():
    plot_id = '09'
    n = 50
    cols = 4
    target_min = 5.0
    var_ids = range(1, 14)
    reg_confs = [RegConf(nam_y=f'cont{i}', nam_x='target') for i in var_ids]

    def fs(v: float) -> int:
        a = int(math.floor(v))
        if a > (n - 1):
            return n - 1
        else:
            return a

    def rows(cnt: int) -> int:
        return int(math.ceil(float(cnt) / cols))

    def plot_contour(conf: RegConf, plot_cnt: int, plot_idx: int,
                     df: pd.DataFrame):
        values_x = df[conf.nam_y].values
        values_y = df[conf.nam_x].values

        value_pairs = zip(values_x, values_y)

        erg = dict([(k, len(list(group))) for k, group in
                    groupby(sorted(value_pairs))])
        z = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                z[i][j] = erg.get((i, j), 0)

        max_cnt = np.amax(z)
        z = np.sqrt(z / max_cnt)
        x = (np.arange(0, n).astype(float) / n) * (
                    10.26 - target_min) + target_min
        y = np.arange(0, n).astype(float) / n

        rs = rows(plot_cnt)
        print("subplot", rs, cols, plot_idx)
        """
        cmaps: ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        """
        ax = plt.subplot(rs, cols, plot_idx)
        # ax.contourf(x, y, z, cmap='Spectral', antialiased=True, levels=30)
        ax.contour(x, y, z, cmap='PuOr', antialiased=True, levels=20)
        ax.set_xlabel(conf.nam_x)
        ax.set_ylabel(conf.nam_y)
        ax.axvline(7.94, color='g')

    scaler = MinMaxScaler(feature_range=(0, n))
    df_train = files.train_df()
    df_train = df_train.loc[df_train['target'] >= target_min]

    scaler.fit(df_train)
    array_transf = scaler.transform(df_train)
    df_scaled = pd.DataFrame(array_transf, columns=df_train.keys()).applymap(fs)

    cfg_cnt = len(reg_confs)
    plt.figure(figsize=(20, 17))
    for idx, reg_conf in zip(range(1, cfg_cnt + 1), reg_confs):
        plot_contour(reg_conf, cfg_cnt, idx, df_scaled)
    plt.suptitle('target against predictors', fontsize=36)
    fnam = files.plotdir / f"regplot_{plot_id}.png"
    plt.savefig(fnam)
    print(f"Plotted to {fnam.absolute()}")


def analyse_target():
    def plot_it(target_values: np.ndarray, plot_id: str, ma_displ: float):
        def f(v: float) -> int:
            if v < 4.0:
                return 1
            else:
                return 0

        r = np.linspace(0, ma_displ, num=100)

        def sc(rv: float) -> int:
            scv = 0
            for xv in target_values:
                if xv < rv:
                    scv += 1
            return scv

        res = [(rv, sc(rv)) for rv in r]

        for x, y in res:
            print(f"{x:7.2f} - {y:7d}")

        plot_dir = files.plotdir
        nam = f"distr_target_{plot_id}.png"
        fnam = plot_dir / nam
        x = [t[0] for t in res]
        y = [t[1] for t in res]

        plt.plot(x, y)
        plt.title(f"cumulative distribution / target max={ma_displ:.2f} ")
        plt.xlabel("target")
        plt.ylabel("cnt")

        plt.savefig(fnam)
        print(f"Plotted to {fnam.absolute()}")

    df = files.train_df()
    target = df.target.values
    max_val = 10.267568500800396
    plot_it(target, "detail", 6.0)
    plot_it(target, "all", max_val)


if __name__ == '__main__':
    contour_all()
