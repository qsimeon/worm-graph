#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _utils.py
@time: 2023/2/28 12:15
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

def neuro_plot(y, isTarget):
    y_np = pd.DataFrame(y)
    # print(y_np.shape)

    # data normalization: z-scoring
    cnt = 0
    interval = 3
    for i in range(0, y_np.shape[0]):
        y_np.iloc[i] = (y_np.iloc[i] - y_np.iloc[i].mean()) / y_np.iloc[i].std()
        y_np.iloc[i] = y_np.iloc[i] + cnt
        cnt -= interval

    # start plotting
    plt.figure(figsize=(6, 12))
    axe = plt.gca()
    axe.spines['top'].set_color('none')
    axe.spines['right'].set_color('none')
    axe.spines['left'].set_color('none')

    # transfer to list in order to fit in ticks-reformat
    list_y = []
    list_label = []

    for j in range(0, y_np.shape[0]):
        list_y.append(-j*interval)
        list_label.append(j)

    plt.ylabel("Status")
    plt.xlabel("Time(s)")

    plt.yticks(list_y, list_label, fontproperties='Times New Roman', size=6)

    for i in range(0, y_np.shape[0]):
        plt.plot(range(0, y_np.shape[1]), y_np.iloc[i], color=sns.color_palette("deep", n_colors=20)[i % 20],
                 linewidth=0.5)

    if isTarget == True:
        plt.savefig('./worm_response_target.png', dpi=1000, bbox_inches='tight')
    else:
        plt.savefig('./worm_response_pred.png', dpi=1000, bbox_inches='tight')
    plt.show()



def derivative(y, t):
    yrow, ycol = y.size()
    dy = np.zeros((yrow-1, ycol))
    for i in range(0, yrow-1):
        dy[i, :] = y[i+1, :] - y[i, :]
    return dy


def poolData(yin, nVars, polyorder, usesine):
    n = yin.shape[0]
    yout = np.zeros((n, 1))

    ind = 0
    # poly order 0
    yout[:,ind] = np.ones(n)
    ind += 1

    # poly order 1
    for i in range(nVars):
        yout = np.c_[yout, np.ones(n)]
        yout[:,ind] = yin[:,i]
        ind += 1

    if polyorder >= 2:
        # poly order 2
        for i in range(nVars):
            for j in range(i, nVars):
                yout = np.c_[yout, np.ones(n)]
                yout[:,ind] = yin[:,i] * yin[:,j]
                ind += 1

    if polyorder >= 3:
        # poly order 3
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    yout = np.c_[yout, np.ones(n)]
                    yout[:,ind] = yin[:,i] * yin[:,j] * yin[:,k]
                    ind += 1

    if polyorder >= 4:
        # poly order 4
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        yout = np.c_[yout, np.ones(n)]
                        yout[:,ind] = yin[:,i] * yin[:,j] * yin[:,k] * yin[:,l]
                        ind += 1

    if polyorder >= 5:
        # poly order 5
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        for m in range(l, nVars):
                            yout = np.c_[yout, np.ones(n)]
                            yout[:,ind] = yin[:,i] * yin[:,j] * yin[:,k] * yin[:,l] * yin[:,m]
                            ind += 1

    if usesine:
        for k in range(1, 11):
            yout = np.concatenate((yout, np.sin(k*yin), np.cos(k*yin)), axis=1)

    return yout


def sparsifyDynamics(Theta, dXdt, lam, n):
    # Compute Sparse regression: sequential least squares
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]  # initial guess: Least-squares

    # lambda is our sparsification knob.
    for k in range(10):
        smallinds = (np.abs(Xi) < lam)  # find small coefficients
        Xi[smallinds] = 0  # and threshold
        for ind in range(n):  # n is state dimension
            biginds = ~smallinds[:, ind]
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]

    return Xi


def sparseGalerkin(y, t, ahat, polyorder, usesine):
    y_ = y.reshape((len(y), 1)).T
    yPool = poolData(y_, len(y), polyorder, usesine)
    dy = np.dot(yPool, ahat).T
    dy = dy.reshape((len(dy), ))
    return dy


def governingFuncPredict(x0, Theta, Xi):
    x_hat = np.dot(Theta, Xi)
    # print(x_hat.shape, "---")
    # print(x_hat.shape)
    pred = calculas(x0.T, x_hat)
    return pred


def calculas(y0, y_hat):
    # this is the reverse of derivative
    # print(y0.shape, y_hat.shape) # [1:3], [301:3]
    yrow, ycol = y_hat.shape[0], y_hat.shape[1]
    sum_y = np.zeros((yrow+1, ycol))
    sum_y[0, :] = y0
    for i in range(1, yrow+1):
        sum_y[i, :] = sum_y[i-1, :] + y_hat[i-1, :]
    return sum_y