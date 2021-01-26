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
default_train_border = 7.94
min_data = 5.0
files = Files()


class SplitModels:
    @staticmethod
    def tuple_model(x: np.ndarray, y: np.ndarray) -> MyModel:
        class M(MyModel):
            model_left: MyModel
            model_right: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data,
                                                            default_train_border)
                self.model_left = Train.fit_gbm(xl, yl,
                                                Train.gbm_optimal_config)
                self.model_right = Train.fit_gbm(xr, yr,
                                                 Train.gbm_optimal_config)

            def predict(self, x_test: np.ndarray) -> np.ndarray:
                pl = self.model_left.predict(x_test)
                pr = self.model_right.predict(x_test)
                print("pl", pl.shape)
                print("pr", pr.shape)
                return np.maximum(pl, pr)

        return M(x, y)

    @staticmethod
    def triple_model_train_border_9_0(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm, train_border=9.0)

    @staticmethod
    def triple_model_train_border_8_5(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm, train_border=8.5)

    @staticmethod
    def triple_model_train_border_8_0(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm, train_border=8.0)

    @staticmethod
    def triple_model_train_border_7_5(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm, train_border=7.5)

    @staticmethod
    def triple_model_train_border_7_0(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm, train_border=7.0)

    @staticmethod
    def triple_model_train_border_6_5(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm, train_border=6.5)

    @staticmethod
    def triple_model_maximum(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 0, 10)
        return SplitModels._triple_model(x, y, np.maximum, cm)

    @staticmethod
    def triple_model_maximum_narrow_l(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 6.5, 9.0)
        return SplitModels._triple_model(x, y, np.maximum, cm)

    @staticmethod
    def triple_model_maximum_narrow_xl(x: np.ndarray, y: np.ndarray) -> MyModel:
        cm = lambda xd, yd: Util.cut_middle(xd, yd, 5.5, 9.5)
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
    def _triple_model(x: np.ndarray, y: np.ndarray, combine: Callable,
                      cut_middle: Callable,
                      train_border: float = 7.94) -> MyModel:
        class M(MyModel):
            model_all: MyModel
            model_left: MyModel
            model_right: MyModel

            def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
                xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data,
                                                            train_border)
                xm, ym = cut_middle(x_data, y_data)
                self.model_all = Train.fit_gbm(xm, ym, Train.gbm_optimal_config)
                self.model_left = Train.fit_gbm(xl, yl,
                                                Train.gbm_optimal_config)
                self.model_right = Train.fit_gbm(xr, yr,
                                                 Train.gbm_optimal_config)

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
                self.model = Train.fit_gbm(x_data, y_data,
                                           Train.gbm_optimal_config)

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
    xl, xr, yl, yr = Util.split_arrays_by_value(x_data, y_data,
                                                default_train_border)
    hist(y_data, 'all', f'target values', color='r')
    hist(yl, 'left', f'target values smaller {default_train_border:.2f}',
         color='g')
    hist(yr, 'right', f'target values greater {default_train_border:.2f}',
         color='b')


def hist_predictions(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,
                                                        random_state=203842039)
    xl, xr, yl, yr = Util.split_arrays_by_value(x_train, y_train,
                                                default_train_border)
    model_all = Train.fit_gbm(x_train, y_train, Train.gbm_optimal_config)
    model_left = Train.fit_gbm(xl, yl, Train.gbm_optimal_config)
    model_right = Train.fit_gbm(xr, yr, Train.gbm_optimal_config)

    yp_all = model_all.predict(x_test)
    yp_left = model_left.predict(x_test)
    yp_right = model_right.predict(x_test)

    hist(yp_all, 'pred_all', f'predicted values', color='orange')
    hist(yp_left, 'pred_left',
         f'predicted values smaller {default_train_border:.2f}', color='orange')
    hist(yp_right, 'pred_right',
         f'predicted values greater {default_train_border:.2f}', color='orange')


@dataclass
class SplitTrain:
    desc: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    f: Callable


def process_split_train(split_train: SplitTrain) -> (str, float):
    return split_train.desc, Train.trainit(split_train.seed, split_train.x,
                                           split_train.y, split_train.f, False)


def run_train_it():
    cnt = 20
    tid = '05'

    def train_it(x_dat, y_dat):
        split_train_cfgs = {
            '01': [
                SplitTrain("no split", 1217, x_dat, y_dat,
                           SplitModels.no_split),
                SplitTrain("triple max", 1983, x_dat, y_dat,
                           SplitModels.triple_model_maximum),
            ],
            '02': [
                SplitTrain("triple mean of g", 1283, x_dat, y_dat,
                           SplitModels.triple_model_mean_of_greatest),
            ],
            '03': [
                SplitTrain("triple cut m", 1281113, x_dat, y_dat,
                           SplitModels.triple_model_maximum_narrow_m),
                SplitTrain("triple cut s", 1232823, x_dat, y_dat,
                           SplitModels.triple_model_maximum_narrow_s),
                SplitTrain("triple cut xs", 145453, x_dat, y_dat,
                           SplitModels.triple_model_maximum_narrow_xs),
            ],
            '04': [
                SplitTrain("triple cut l", 132823, x_dat, y_dat,
                           SplitModels.triple_model_maximum_narrow_l),
                SplitTrain("triple cut xl", 54453, x_dat, y_dat,
                           SplitModels.triple_model_maximum_narrow_xl),
            ],
            '05': [
                SplitTrain(
                    "border 6.5", 54445, x_dat, y_dat,
                    SplitModels.triple_model_train_border_6_5),
                SplitTrain(
                    "border 7.0", 54458, x_dat, y_dat,
                    SplitModels.triple_model_train_border_7_0),
                SplitTrain(
                    "border 7.5", 54485, x_dat, y_dat,
                    SplitModels.triple_model_train_border_7_5),
                SplitTrain(
                    "border 8.0", 54445, x_dat, y_dat,
                    SplitModels.triple_model_train_border_8_0),
                SplitTrain(
                    "border 8.5", 54412, x_dat, y_dat,
                    SplitModels.triple_model_train_border_8_5),
                SplitTrain(
                    "border 9.0", 12445, x_dat, y_dat,
                    SplitModels.triple_model_train_border_9_0),
                SplitTrain(
                    "border best", 823, x_dat, y_dat,
                    SplitModels.triple_model_maximum),
                SplitTrain(
                    "no split", 1217, x_dat, y_dat,
                    SplitModels.no_split),
            ],
        }

        split_trains = split_train_cfgs[tid]
        for st in split_trains:
            with multiprocessing.Pool() as pool:
                np.random.seed(st.seed)
                seeds = np.random.randint(0, 1000000, cnt)
                sts = [SplitTrain(desc=st.desc, seed=s, x=st.x, y=st.y, f=st.f)
                       for s in seeds]
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
        title: str
        data: dict

    cfgs = {
        'triple': Cfg(
            title='compare split / no split',
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
            }
        ),
        'mog': Cfg(
            title='split - mean of great',
            data={
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
                'triple mean of great': [
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
        ),
        'cutm': Cfg(
            title='cut size',
            data={
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
                'triple cut xs': [
                    0.7328030908142162,
                    0.7338235540906083,
                    0.7341963237764203,
                    0.733923839938995,
                    0.7332010542057551,
                    0.7324815669108985,
                    0.7337413003517282,
                    0.7351442215591371,
                    0.7331096753788353,
                    0.7319045088032006,
                    0.7328750829983506,
                    0.7329963269956343,
                    0.7323883509953255,
                    0.7329364701778425,
                    0.7347750965284029,
                    0.7326771315762366,
                    0.7351769017143668,
                    0.7322016004299307,
                    0.733544450701847,
                    0.7353707429350909,
                ],
                'triple cut s': [
                    0.7341656663707038,
                    0.7310818855164775,
                    0.7324955209475246,
                    0.7330198504183878,
                    0.7327176966044778,
                    0.7342309420412005,
                    0.7342330791765426,
                    0.7333652188250711,
                    0.7351713416220367,
                    0.7319503147669266,
                    0.7327750941082221,
                    0.731515534167976,
                    0.7343650056456157,
                    0.7357654741028172,
                    0.7323856767265006,
                    0.7324586685337023,
                    0.7352860304351114,
                    0.7338738585832775,
                    0.7355666347865094,
                    0.7353088546956907,
                ],
                'triple cut m': [
                    0.7296870013626504,
                    0.7327805696254299,
                    0.7316794175658746,
                    0.7301526429495326,
                    0.7305282799804192,
                    0.7310194446098214,
                    0.729513334403088,
                    0.7307882477422737,
                    0.7279550393413717,
                    0.728794434102106,
                    0.7286725459007971,
                    0.7294277860181257,
                    0.7313543617163364,
                    0.7301752228400468,
                    0.7292749795423806,
                    0.729487844172261,
                    0.7276232279701352,
                    0.729781970898164,
                    0.7301975350838626,
                    0.7290644319642661,
                ],
                'triple cut l': [
                    0.7091952534280209,
                    0.7104384053116533,
                    0.711958514491844,
                    0.7090374421789568,
                    0.7109989630756796,
                    0.7099821080902462,
                    0.7109541898677125,
                    0.7096217398378141,
                    0.7104492786113709,
                    0.708727276549688,
                    0.7097606683042734,
                    0.7095248451346012,
                    0.7093506118106352,
                    0.70917151926084,
                    0.7088937709388792,
                    0.710070759868951,
                    0.7095874678284344,
                    0.7112584791286697,
                    0.7094860968379312,
                    0.7099695072295946,
                ],
                'triple cut xl': [
                    0.7045609620915083,
                    0.7045245747674157,
                    0.7030445261397842,
                    0.7042623103591629,
                    0.7029343588955506,
                    0.704638461317319,
                    0.7023062628349683,
                    0.7053937582933063,
                    0.7044995536372146,
                    0.7052142877498144,
                    0.7044228359297706,
                    0.7043450069642373,
                    0.7020167769591862,
                    0.7030799911428479,
                    0.7049007853338684,
                    0.7046828269484198,
                    0.7028509538032434,
                    0.7043124087042378,
                    0.704670130443317,
                    0.7023655404839197,
                ],
            }
        ),
        'moving_border': Cfg(
            title='moving border',
            data={
                'no split': [
                    0.7014148844484943,
                    0.7033059728622219,
                    0.701431903933839,
                    0.7003701836520039,
                    0.6996296440153382,
                    0.7036599022434028,
                    0.7031841773484276,
                    0.7025788294646532,
                    0.7028740095449288,
                    0.7034649769645311,
                    0.7024103657432053,
                    0.7031046670550005,
                    0.7028793269520328,
                    0.7020930258980881,
                    0.7011240014874999,
                    0.702816716322386,
                    0.7022612371427334,
                    0.701212374742002,
                    0.7007296774521992,
                    0.7017519057964343],
                'border best': [
                    0.7024171181506493,
                    0.7012499557081038,
                    0.7029209783510816,
                    0.7026683877066054,
                    0.7002148283843418,
                    0.7015898535004972,
                    0.7025766180781373,
                    0.7018594156222477,
                    0.7005519493775845,
                    0.7013794709838519,
                    0.7025836736349869,
                    0.7032246848773078,
                    0.7016902682432196,
                    0.7019010452553714,
                    0.7018347979900061,
                    0.701034634751626,
                    0.7015667252841227,
                    0.7013105128333722,
                    0.7024781647495273,
                    0.7029138191494518],
                'border 9.0': [
                    0.7006829995237104,
                    0.7012420349247196,
                    0.7036073680475045,
                    0.7014300760013586,
                    0.6999026738026297,
                    0.7021689299537321,
                    0.7033424951404307,
                    0.7024066295724732,
                    0.7012628143164316,
                    0.7024527557126367,
                    0.7008000691288977,
                    0.7019829022411256,
                    0.7036874563890371,
                    0.7013807657728324,
                    0.7003139988428322,
                    0.7024412243681413,
                    0.7004207051240013,
                    0.7025263088944966,
                    0.7008726517003951,
                    0.700919995994433],
                'border 8.5': [
                    0.7035589072918949,
                    0.7011206155490713,
                    0.7018221925403919,
                    0.7019992637135523,
                    0.7030895880763276,
                    0.7021686656673524,
                    0.7014603264265934,
                    0.7027622498131127,
                    0.6989969439001151,
                    0.7019664923339739,
                    0.702094893498707,
                    0.7003405033614227,
                    0.7001019880271386,
                    0.7027859320947608,
                    0.7024420083275629,
                    0.7018880277930976,
                    0.7017589866697646,
                    0.7019829735187455,
                    0.6997486591653737,
                    0.7009054760318107],
                'border 8.0': [
                    0.7009500652124475,
                    0.702724205884801,
                    0.7008994021014461,
                    0.7022513368408668,
                    0.7031688799334002,
                    0.7008200258808583,
                    0.7008953740525118,
                    0.7045745897952371,
                    0.7024027050577039,
                    0.7014185487881356,
                    0.704007514908653,
                    0.7012372383157974,
                    0.7004370023179909,
                    0.7025959184588509,
                    0.700692398721449,
                    0.700982978719751,
                    0.7038048824097621,
                    0.7041354537336404,
                    0.7004614140911186,
                    0.7030091768084752],
                'border 7.5': [
                    0.7020420995672005,
                    0.7026147208381157,
                    0.7010074558608578,
                    0.7016988747972817,
                    0.7021657167993991,
                    0.7021323370856518,
                    0.7025401987799931,
                    0.7026142349064992,
                    0.7017312324625029,
                    0.7007328867762465,
                    0.6994348394243916,
                    0.700866405807477,
                    0.7026949806239652,
                    0.7029564704760993,
                    0.7012699798235945,
                    0.7020234535185448,
                    0.7012018437486863,
                    0.7010240360411117,
                    0.7015755424302791,
                    0.7038839874730164],
                'border 7.0': [
                    0.7019979003615928,
                    0.701202543726174,
                    0.7024535071886835,
                    0.7028124510617201,
                    0.7040134936043513,
                    0.7017409136380521,
                    0.7020177324347266,
                    0.7025702246312909,
                    0.7027042417675596,
                    0.7013350629852197,
                    0.7011883636060783,
                    0.7022294588557405,
                    0.7010958664268493,
                    0.7022310421872108,
                    0.7037356152050992,
                    0.7027822542185829,
                    0.7012602994762515,
                    0.7020810993835976,
                    0.7022354340524191,
                    0.7008913076669895],
                'border 6.5': [
                    0.7009201040035252,
                    0.7027081673110559,
                    0.7008581196720387,
                    0.7022947175719573,
                    0.7032137689177376,
                    0.7009281166384933,
                    0.7009184582436597,
                    0.7045286150931631,
                    0.7023504960927,
                    0.7014730467334302,
                    0.7040934267804045,
                    0.7012526591072944,
                    0.7004448468839677,
                    0.7026033185180813,
                    0.7006759541808313,
                    0.7009614591033544,
                    0.7038358875755991,
                    0.7041170596147399,
                    0.7005555315370599,
                    0.703018052463878],
            }
        ),
    }
    for pid in cfgs.keys():
        plt.clf()
        cfg = cfgs[pid]
        plt.figure(figsize=(15, 12))
        plt.xticks(rotation=45)
        plt.title(cfg.title)
        plt.boxplot(cfg.data.values(), labels=cfg.data.keys())
        plot_dir = files.plotdir
        nam = f"splittarget_result_{pid}.png"
        fnam = plot_dir / nam
        plt.savefig(fnam)
        print("wrote splittarget result to", fnam.absolute())


def main():
    # run_hists()
    # run_train_it()
    run_boxplot()


if __name__ == '__main__':
    main()
