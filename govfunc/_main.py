#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _main.py
@time: 2023/2/28 12:15
'''

from data._main import *
from govfunc._utils import *

from scipy.integrate import odeint
import matplotlib.pyplot as plt


def main():
    config = OmegaConf.load("./dataset.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)
    print("----dataset prepared------")

    # take three worms as an example
    numOfWorm = 3
    worm = []

    for i in range(0, numOfWorm):
        worm.append(dataset["worm" + str(i)])
        worm[i]["calcium_data"] = worm[i]["calcium_data"].T
        # print(worm[i]["calcium_data"].shape)
    rows, cols = worm[0]["calcium_data"].size()
    print("the time steps is " + str(rows))  # 302

    dys = []
    for i in range(0, numOfWorm):
        dy = derivative(worm[i]["calcium_data"], 0)
        # print(dy.shape)
        dys.append(dy)


    ##########################################
    # here we just use one worm
    x = worm[0]["calcium_data"]
    dx = dys[0]

    slices = 50
    x = x[:, 0:slices]
    dx = dx[:, 0:slices]
    print("x_initial: ", x.shape)
    print("dx: ", dx.shape)


    # the original status x0
    x0 = x[0]
    print("x0: ", x0.shape)

    polyorder = 3
    usesine = False
    r, c = x.shape
    print("r, c", r, c)
    Theta = poolData(x[1:], c, polyorder, usesine)
    print("Theta: ", Theta.shape)

    lambda_ = 0.025
    Xi = sparsifyDynamics(Theta, dx, lambda_, c)
    print("Xi: ", Xi.shape)

    # true calcium_data: worm[0]["calcium_data"]
    # denoted as target
    target = x
    target = target.T
    tr, tc = target.size()
    print("the shape of target: (" + str(tr) + ", " + str(tc) + ")")

    # model's results - generated from governing functions
    # denoted as pred

    pred = governingFuncPredict(x0, Theta, Xi)
    pred = torch.tensor(pred)
    pred = pred.T
    tr, tc = pred.size()
    print("the shape of pred: (" + str(tr) + ", " + str(tc) + ")")

    neuro_plot(target[0:20, :], True)
    neuro_plot(pred[0:20, :], False)

    return 0


if __name__ == "__main__":
    main()
