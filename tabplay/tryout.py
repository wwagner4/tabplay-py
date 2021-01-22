import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabplay import Files, Train, Util

util = Util()
files = Files()


def scaler():
    x = [
        [0, 0],
        [3, 0],
        [1, 1],
        [1, 2],
    ]
    x1 = [
        [0, 0],
        [3, 0],
        [2, 5],
        [1, 1],
    ]
    y = [
        [0],
        [3],
        [1],
        [1],
    ]
    y1 = [
        [0],
        [3],
        [2],
        [2],
    ]
    xscaler = StandardScaler()
    yscaler = StandardScaler()
    xscaler.fit(x)
    yscaler.fit(y)
    a = xscaler.transform(x1)
    pprint(a)
    b = xscaler.inverse_transform(a)
    pprint(b)
    a1 = yscaler.transform(y1)
    pprint(a1)
    b1 = yscaler.inverse_transform(a1)
    pprint(b1)


def split():
    def small():
        x = np.arange(10).reshape((5, 2))
        y = np.array(range(5)).reshape(5, 1)
        print(x.shape, type(x))
        print(y.shape, type(y))
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
        print("- train")
        print(x_train)
        print(y_train)
        print("- test")
        print(x_test)
        print(y_test)

    def large():
        t = files.train_df()
        print("t", type(t))
        a, b = train_test_split(t, test_size=0.3)
        print("a,b", type(a), type(b), a.shape, b.shape)

    small()
    large()


def gbm():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    train = Train()

    train_df = files.train_df().head(n=20000)

    x = train_df[train.x_names].values
    y = train_df[[train.y_name]].values.ravel()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=0)
    reg = GradientBoostingRegressor(random_state=0)
    print("calling fit for gbm")
    print("fit", reg.fit(x_train, y_train))
    print("predict", reg.predict(x_test[1:2]))
    print("score", reg.score(x_test, y_test))


def argparse_tryout():
    cdict = {
        "01": "something 01",
        "02": "something 02"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("id", choices=cdict.keys(), help="The id to run")
    myargs: argparse.Namespace = parser.parse_args()
    print("myargs", myargs)
    print("myargs id", myargs.id)
    print("myargs id", type(myargs.id))
    print("dict val", cdict[myargs.id])


def plot_tryout():
    files = Files()
    results = [
        ('A', 0.71),
        ('B', 0.70),
        ('B1', 0.743),
        ('C', 0.733),
        ('default', 0.731),
        ('all', 0.723),
    ]

    nam = f"tryout_plot.png"
    plot_dir = files.workdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()
    fnam = plot_dir / nam
    all_data = [r[1] for r in results]
    all_labels = [r[0] for r in results]
    plt.ylim(0.69, 0.75)
    plt.title("Tryout plot")
    plt.axhline(0.699, color='r')
    plt.axhline(0.7013, color='g')
    plt.plot(all_labels, all_data)

    plt.savefig(fnam)
    print(f"Plotted to {fnam.absolute()}")


def surface_tryout():
    x = np.arange(0, 10)
    y = np.arange(0, 10)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.sin(r)
    print("--X", x)
    print("--Y", y)
    print("--Z", z)


def np_sel_rows():
    a = np.array([True, True, True, False, False])
    b = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T
    e = b[a, :]
    print(e)


def np_split_x_y():
    rows = 5
    cols = 3

    yv = np.random.random(rows) * 10
    y = yv.reshape(len(yv), 1)
    x = np.random.random(rows * cols).reshape(rows, cols)

    print("x shape", x.shape)
    print("y shape", y.shape)
    print("x", x)
    print("y", y)

    a, b, c, d = util.split_arrays_by_value(x, y, 2.0)
    print("a", a)
    print("b", b)
    print("c", c)
    print("d", d)


if __name__ == '__main__':
    np_split_x_y()
