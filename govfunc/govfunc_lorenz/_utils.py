#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.1
@file: _utils.py
@time: 2023/2/28 10:57
@desc:
"""

import numpy as np


def lorenz(y, t, sigma, beta, rho):
    dy = np.zeros_like(y)
    dy[0] = sigma * (y[1] - y[0])
    dy[1] = y[0] * (rho - y[2]) - y[1]
    dy[2] = y[0] * y[1] - beta * y[2]
    return dy


def poolData(yin, nVars, polyorder, usesine):
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


def sparseGalerkin(y, t, ahat, polyorder, usesine):
    y_ = y.reshape((len(y), 1)).T
    yPool = poolData(y_, len(y), polyorder, usesine)
    dy = np.dot(yPool, ahat).T
    dy = dy.reshape((len(dy),))
    return dy
