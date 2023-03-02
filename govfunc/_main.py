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
    print("----dataset prepared------\n")

    # take three worms as an example
    numOfWorm = 3
    worm = []

    for i in range(0, numOfWorm):
        worm.append(dataset["worm" + str(i)])
        worm[i]["calcium_data"] = worm[i]["calcium_data"]
        # print(worm[i]["calcium_data"].shape)
    rows, cols = worm[0]["calcium_data"].size()
    print("the time step is " + str(rows))

    dys = []
    for i in range(0, numOfWorm):
        dy = derivative(worm[i]["calcium_data"], 0)
        # print(dy.shape)
        dys.append(dy)

    ##########################################
    # here we just use one worm
    x = worm[0]["calcium_data"]
    dx = derivative(x, 0)

    # dataset is sliced based on number of neurons(parameter: slices) because of the huge amount of parameters
    slices = 30
    x = x[:, 0:slices]
    dx = dx[:, 0:slices]
    # print("x_initial: ", x.shape)
    # print("dx: ", dx.shape)

    # the original status x0 from time step 0
    x0 = x[0]
    # print("x0: ", x0.shape)

    # polyorder: polynomial formula with the first variant up to x^(i), where i is chosen from [1, polyorder]
    polyorder = 4
    usesine = False
    r, c = x.shape
    Theta = poolData(x[1:], c, polyorder, usesine)
    print("Theta: ", Theta.shape)

    # TODO: figure out the influence of changing lambda_
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

    time_slice = 1000
    neuro_plot(target[:, 0:time_slice], True)
    neuro_plot(pred[:, 0:time_slice], False)

    return 0


if __name__ == "__main__":
    main()
