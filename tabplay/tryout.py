from pprint import pprint

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabplay import Files, Train


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
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        print("- train")
        print(x_train)
        print(y_train)
        print("- test")
        print(x_test)
        print(y_test)

    def large():
        files = Files()
        t = files.train_df()
        print("t", type(t))
        a, b = train_test_split(t, test_size=0.3)
        print("a,b", type(a), type(b), a.shape, b.shape)

    large()

def gbm():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    files = Files()
    train= Train()

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


gbm()
