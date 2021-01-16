import math
from dataclasses import dataclass
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler

from tabplay import Files


@dataclass
class RegConf:
    nam_x: str
    nam_y: str


"""
cmaps: ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
Index(['id', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
       'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14',
       'target'],
"""
files = Files()
plot_id = '04'
n = 50
cols = 4
cmap = 'coolwarm'
figure(figsize=(20, 17))
var_ids = range(1, 14)

reg_confs = [RegConf(nam_x=f'cont{i}', nam_y='target') for i in var_ids]


def fs(v: float) -> int:
    a = int(math.floor(v))
    if a > (n - 1):
        return n - 1
    else:
        return a


def rows(cnt: int) -> int:
    return int(math.ceil(float(cnt) / cols))


def plot1(reg_conf: RegConf, plot_cnt: int, plot_idx: int, df_scaled: pd.DataFrame):
    values_x = df_scaled[reg_conf.nam_x].values
    values_y = df_scaled[reg_conf.nam_y].values

    value_pairs = zip(values_x, values_y)

    erg = dict([(k, len(list(group))) for k, group in groupby(sorted(value_pairs))])
    z = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            z[i][j] = erg.get((i, j), 0)

    max_cnt = np.amax(z)
    z = np.sqrt(z / max_cnt)
    x = np.arange(0, n).astype(float) / n
    y = np.arange(0, n).astype(float) / n

    rs = rows(plot_cnt)
    print("subplot", rs, cols, plot_idx)
    ax = plt.subplot(rs, cols, plot_idx)
    ax.contourf(x, y, z, cmap=cmap, antialiased=True, levels=15)
    ax.set_xlabel(reg_conf.nam_x)
    ax.set_ylabel(reg_conf.nam_y)


def main():
    scaler = MinMaxScaler(feature_range=(0, n))
    df = files.train_df()
    scaler.fit(df)
    array_transformed = scaler.transform(df)
    df_scaled = pd.DataFrame(array_transformed, columns=df.keys()).applymap(fs)

    cfg_cnt = len(reg_confs)
    for idx, reg_conf in zip(range(1, cfg_cnt + 1), reg_confs):
        plot1(reg_conf, cfg_cnt, idx, df_scaled)
    plt.suptitle('predictors against target', fontsize=16)
    fnam = files.plotdir / f"regplot_{plot_id}.png"
    plt.savefig(fnam)
    print(f"Plotted to {fnam.absolute()}")


if __name__ == '__main__':
    main()
