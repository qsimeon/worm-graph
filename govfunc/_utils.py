#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _utils.py
@time: 2023/2/28 12:15
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def generate_polynomial(x, polyorder, usesine):
    # polyorder: polynomial formula with the first variant up to x^(i), where i is chosen from [1, polyorder]
    r, c = x.shape
    Theta = poolData(x, c, polyorder, usesine)
    return Theta

def neuro_plot(y, isTarget):
    y_df = pd.DataFrame(y)
    # data normalization: z-scoring
    cnt = 0
    interval = 10
    for i in range(0, y_df.shape[0]):
        y_df.iloc[i] = (y_df.iloc[i] - y_df.iloc[i].mean()) / y_df.iloc[i].std()
        y_df.iloc[i] = y_df.iloc[i] + cnt
        cnt -= interval
    # start plotting
    plt.figure(figsize=(6, 12))
    axe = plt.gca()
    axe.spines["top"].set_color("none")
    axe.spines["right"].set_color("none")
    axe.spines["left"].set_color("none")

    # transfer to list in order to re-label the y-axis
    list_y = []
    list_label = []

    for j in range(0, y_df.shape[0]):
        list_y.append(-j * interval)
        list_label.append(j)

    plt.ylabel("Neurons")
    plt.xlabel("Time(s)")

    plt.yticks(list_y, list_label, fontproperties="Times New Roman", size=6)

    for i in range(0, y_df.shape[0]):
        plt.plot(
            range(0, y_df.shape[1]),
            y_df.iloc[i],
            color=sns.color_palette("deep", n_colors=20)[i % 20],
            linewidth=0.5,
        )

    if isTarget == True:
        plt.savefig("./worm_response_target.png", dpi=1000, bbox_inches="tight")
    else:
        plt.savefig("./worm_response_pred.png", dpi=1000, bbox_inches="tight")
    plt.show()


def derivative(y, t):
    """
    input: [time, status]
    func: calculate the residual between time steps
    output: [residual(\delta t), status]
    """
    yrow, ycol = y.size()
    dy = np.zeros((yrow - 1, ycol))
    for i in range(0, yrow - 1):
        dy[i, :] = y[i + 1, :] - y[i, :]
    return dy


def poolData(yin, nVars, polyorder, usesine):
    """
    func: generate polynomial functions as candidates
    output: \theta(yin) as denoted in paper
    """
    n = yin.shape[0]
    yout = np.zeros((n, 1))

    ind = 0
    # poly order 0
    yout[:, ind] = np.ones(n)
    ind += 1

    # poly order 1
    for i in range(nVars):
        yout = np.c_[yout, np.ones(n)]
        yout[:, ind] = yin[:, i]
        ind += 1

    if polyorder >= 2:
        # poly order 2
        for i in range(nVars):
            for j in range(i, nVars):
                yout = np.c_[yout, np.ones(n)]
                yout[:, ind] = yin[:, i] * yin[:, j]
                ind += 1

    if polyorder >= 3:
        # poly order 3
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    yout = np.c_[yout, np.ones(n)]
                    yout[:, ind] = yin[:, i] * yin[:, j] * yin[:, k]
                    ind += 1

    if polyorder >= 4:
        # poly order 4
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        yout = np.c_[yout, np.ones(n)]
                        yout[:, ind] = yin[:, i] * yin[:, j] * yin[:, k] * yin[:, l]
                        ind += 1

    if polyorder >= 5:
        # poly order 5
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        for m in range(l, nVars):
                            yout = np.c_[yout, np.ones(n)]
                            yout[:, ind] = (
                                yin[:, i]
                                * yin[:, j]
                                * yin[:, k]
                                * yin[:, l]
                                * yin[:, m]
                            )
                            ind += 1

    if usesine:
        for k in range(1, 11):
            yout = np.concatenate((yout, np.sin(k * yin), np.cos(k * yin)), axis=1)

    return yout


def sparsifyDynamics(Theta, dXdt, lam, n):
    """
    func: calculate coefficients of \theta() generated from poolData(...) using dynamic regression
    note: consume large computational resources
    """
    # Compute Sparse regression: sequential least squares
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]  # initial guess: Least-squares

    # lambda is our sparsification knob.
    for k in range(10):
        smallinds = np.abs(Xi) < lam  # find small coefficients
        Xi[smallinds] = 0  # and threshold
        for ind in range(n):  # n is state dimension
            biginds = ~smallinds[:, ind]
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(
                Theta[:, biginds], dXdt[:, ind], rcond=None
            )[0]

    return Xi


def governingFuncPredict(x0, Theta, Xi):
    x_hat = np.dot(Theta, Xi)
    # print(x_hat.shape, "---")
    # print(x_hat.shape)
    pred = calculas(x0, x_hat)
    return pred


def calculas(y0, y_hat):
    """
    this is the reverse of derivative
    """
    # print(y0.shape, y_hat.shape)
    yrow, ycol = y_hat.shape[0], y_hat.shape[1]
    sum_y = np.zeros((yrow + 1, ycol))
    sum_y[0, :] = y0
    for i in range(1, yrow + 1):
        sum_y[i, :] = sum_y[i - 1, :] + y_hat[i - 1, :]
    return sum_y
