import multiprocessing
from dataclasses import dataclass
from pprint import pprint
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from tabplay import Files, Train, Util, MyModel

"""
# @ ben
docker run \
 --detach \
 -v /home/wwagner4/prj/oldschool/tabplay-py:/opt/project \
 -v /data/work/tabplay:/opt/work \
 tabplay \
 python -u /opt/project/tabplay/splittarget.py
"""

train_border: float = 7.94
min_data = 5.0
files = Files()


class SplitModels:
    @staticmethod
    def tuple_model(x: np.ndarray, y: np.ndarray) -> MyModel:
        class M(MyModel):
            model_left: MyModel
            model_right: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data, train_border)
                self.model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
                self.model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                pl = self.model_left.predict(x_test)
                pr = self.model_right.predict(x_test)
                print("pl", pl.shape)
                print("pr", pr.shape)
                return np.maximum(pl, pr)

        return M(x, y)

    @staticmethod
    def triple_model_maximum(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm)

    @staticmethod
    def triple_model_maximum_narrow_m(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 7.4, 8.4)
        return SplitModels._triple_model(x, y, np.maximum, cm)

    @staticmethod
    def triple_model_maximum_narrow_s(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 7.88, 8.0)
        return SplitModels._triple_model(x, y, np.maximum, cm)

    @staticmethod
    def triple_model_maximum_narrow_xs(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 7.92, 7.96)
        return SplitModels._triple_model(x, y, np.maximum, cm)

    @staticmethod
    def triple_model_mean_of_greatest(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, Util.mean_of_greatest, cm)

    @staticmethod
    def _triple_model(x: np.ndarray, y: np.ndarray, combine: Callable, cut_middle: Callable) -> MyModel:
        class M(MyModel):
            model_all: MyModel
            model_left: MyModel
            model_right: MyModel


            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):

                xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data, train_border)
                xm, ym = cut_middle(x_data, y_data)
                self.model_all = Train.fit_gbm(xm, ym, Train.gbm_optimal_config)
                self.model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
                self.model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                pa = self.model_all.predict(x_test)
                pl = self.model_left.predict(x_test)
                pr = self.model_right.predict(x_test)
                print("pa", pa.shape)
                print("pl", pl.shape)
                print("pr", pr.shape)
                return combine(pa, pl, pr)

        return M(x, y)

    @staticmethod
    def no_split(x: np.ndarray, y: np.ndarray) -> MyModel:
        class M(MyModel):
            model: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                self.model = Train.fit_gbm(x_data, y_data, Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                return self.model.predict(x_test)

        return M(x, y)


def hist(data: np.ndarray, hist_id: str, title: str, color: str):
    plt.clf()
    plt.hist(x=data, bins=100, facecolor=color, alpha=0.75)
    plt.ylabel('count')
    plt.xlabel('target')
    plt.title(title)
    plt.axis([5, 10, 0, 10000])
    plt.grid(True)
    plot_dir = Files.plotdir
    nam = f"splittarget_{hist_id}.png"
    fnam = plot_dir / nam
    plt.savefig(fnam)
    print("wrote histogran to", fnam.absolute())


def run_hists():
    df_train = files.train_df()
    x_data = df_train[Train.x_names].values
    y_data = df_train[[Train.y_name]].values
    xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data, train_border)
    hist(y_data, 'all', f'target values', color='r')
    hist(yl, 'left', f'target values smaller {train_border:.2f}', color='g')
    hist(yr, 'right', f'target values greater {train_border:.2f}', color='b')


def hist_predictions(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=203842039)
    xl, xr, yl, yr = Util.split_arrays_by_value(x_train, y_train, train_border)
    model_all = Train.fit_gbm(x_train, y_train, Train.gbm_optimal_config)
    model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
    model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

    yp_all = model_all.predict(x_test)
    yp_left = model_left.predict(x_test)
    yp_right = model_right.predict(x_test)

    hist(yp_all, 'pred_all', f'predicted values', color='orange')
    hist(yp_left, 'pred_left', f'predicted values smaller {train_border:.2f}', color='orange')
    hist(yp_right, 'pred_right', f'predicted values greater {train_border:.2f}', color='orange')


@dataclass
class SplitTrain:
    desc: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    f: Callable


def process_split_train(split_train: SplitTrain) -> (str, float):
    return split_train.desc, Train.trainit(split_train.seed, split_train.x, split_train.y, split_train.f, False)


def run_train_it():
    cnt = 20
    tid = '03'

    def train_it(x_dat, y_dat):
        split_train_cfgs = {
            '01': [
                SplitTrain("no split", 1217, x_dat, y_dat, SplitModels.no_split),
                SplitTrain("triple max", 1983, x_dat, y_dat, SplitModels.triple_model_maximum),
            ],
            '02': [
                SplitTrain("triple mean of g", 1283, x_dat, y_dat, SplitModels.triple_model_mean_of_greatest),
            ],
            '03': [
                SplitTrain("triple cut m", 1281113, x_dat, y_dat, SplitModels.triple_model_maximum_narrow_m),
                SplitTrain("triple cut s", 1232823, x_dat, y_dat, SplitModels.triple_model_maximum_narrow_s),
                SplitTrain("triple cut xs", 145453, x_dat, y_dat, SplitModels.triple_model_maximum_narrow_xs),
            ]
        }
        split_trains = split_train_cfgs[tid]
        for st in split_trains:
            with multiprocessing.Pool() as pool:
                np.random.seed(st.seed)
                seeds = np.random.randint(0, 1000000, cnt)
                sts = [SplitTrain(desc=st.desc, seed=s, x=st.x, y=st.y, f=st.f) for s in seeds]
                result = {}
                for i in pool.map(process_split_train, sts):
                    result.setdefault(i[0], []).append(i[1])
                pprint(result)

    print(f"---> run_train_it id:{tid} cnt:{cnt}")
    df_train = files.train_df()
    x_all = df_train[Train.x_names].values
    y_all = df_train[[Train.y_name]].values
    _, x, _, y = Util.split_arrays_by_value(x_all, y_all, min_data)
    print('x', x.shape)
    print('y', y.shape)

    train_it(x, y)
    print(f"<--- run_train_it id:{tid} cnt:{cnt}")


def run_boxplot():
    @dataclass
    class Cfg:
        data: dict

    cfgs = {
        '01': Cfg(
            data={
                'no_split': [
                    0.7039493389539233,
                    0.7013205916708355,
                    0.7016550741001032,
                    0.7019636532330444,
                    0.7013762482873541,
                ],
                'triple': [
                    0.6997663848174556,
                    0.7029153140637895,
                    0.703757391969722,
                    0.7019180929865728,
                ],
            }
        ),
        '02': Cfg(
            data={
                'no_split': [
                    0.7040340944617459,
                    0.7012621930207075,
                    0.7015999076371211,
                    0.7019887945001956,
                    0.7013959623820328,
                ],
                'triple': [
                    0.702925914913469,
                    0.7038798217630783,
                    0.7019596574480624,
                    0.7025962353500002,
                    0.7002881835337523,
                    0.7038416475633518,
                ],
            }
        ),
        '03': Cfg(
            data={
                'triple': [
                    0.7034574080360404,
                    0.7013682751928109,
                    0.7039666641729853,
                    0.7019976854314919,
                    0.6997011829206079,
                    0.7026926219215773,
                    0.7004161859403136,
                    0.7045940317800061,
                    0.7022860223744184,
                    0.7017464161059427,
                    0.7033206731642525,
                    0.7027561629313278,
                    0.702550820244086,
                    0.7037602289066305,
                    0.7017613795853135,
                    0.701943261631429,
                    0.7041012262447615,
                    0.7020766932988456,
                    0.7025875804429066,
                    0.7025652516936456,
                ],
                'no split': [
                    0.7009199711774573,
                    0.7014282805510063,
                    0.7031982159585766,
                    0.7024588163765331,
                    0.6999652080488968,
                    0.7012884086845848,
                    0.7025801527504894,
                    0.702324032893071,
                    0.7009962556988304,
                    0.7010960945547321,
                    0.7017451433588406,
                    0.7021371837073322,
                    0.7008733939329123,
                    0.7025726175124827,
                    0.7014632972039827,
                    0.7027996873054194,
                    0.7019968076158548,
                    0.7032934889252749,
                    0.7028908903141194,
                    0.7038925429247523,
                ],
                'triple mean of g': [
                    0.7986880745333453,
                    0.7998060779354146,
                    0.7996435960022191,
                    0.7992940746704614,
                    0.7964792340691824,
                    0.7982232977720759,
                    0.7965940251812592,
                    0.7990267085982752,
                    0.7981864832574558,
                    0.7989153431955245,
                    0.7986177923723153,
                    0.7970921731142104,
                    0.7990817981251812,
                    0.7992400347281507,
                    0.7989207047181026,
                    0.7969949945999292,
                    0.7988214870022083,
                    0.8020464176159474,
                    0.7996150270896095,
                    0.79826040414,
                ]
            }
        )
    }
    pid = '03'
    cfg = cfgs[pid]
    plt.boxplot(cfg.data.values(), labels=cfg.data.keys())
    plot_dir = files.plotdir
    nam = f"splittarget_result_{pid}.png"
    fnam = plot_dir / nam
    plt.savefig(fnam)
    print("wrote splittarget result to", fnam.absolute())


def main():
    # run_hists()
    run_train_it()
    # run_boxplot()


if __name__ == '__main__':
    main()
