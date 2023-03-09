#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: smooth_method_trial.py
@time: 2023/3/3 16:36
"""

from diff_tvr import DiffTVR
import numpy as np
import matplotlib.pyplot as plt

from diff_tvr import *
from omegaconf import OmegaConf
from govfunc._utils import *
from data._main import *
from govfunc._utils import *
from numpy.fft import fft
from scipy.signal import savgol_filter
from torch.fft import fft
import torch.nn as nn
import matplotlib.pyplot as plt







if __name__ == "__main__":

    config = OmegaConf.load("../../conf/dataset.yaml")
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
    data = worm[0]["calcium_data"]
    data_torch = torch.tensor(data)


    n = data.shape[0]

    # print(worm[0].keys())
    # print(worm[0]["neuron_to_idx"],'\n', worm[0]["num_neurons"], worm[0]["num_named_neurons"], worm[0]["num_unknown_neurons"])


    # try the basic model
    cols_with_data_mask = worm[0]["neurons_mask"]
    labels_neurons_with_data = [worm[0]["slot_to_neuron"][slot] for slot, boole in enumerate(cols_with_data_mask) if
                                boole.item() is True]

    cal_data = worm[0]["smooth_calcium_data"][:, cols_with_data_mask]


    # how smooth
    plt.plot(cal_data[:, 0])
    plt.title("smooth visualization")
    plt.show()




    exit(0)
    ###################### for the simplest linear combination on neuronal signals #################
    # pred y_t+1 = 2 * y_t - y_t-1
    pred = 2 * cal_data[1:-1] - cal_data[0:-2]
    # baseline: y_t+1 == y_t
    baseline = cal_data[1:-1]
    # target: the real data for y_t+1
    target = cal_data[2:]
    print(target.shape, pred.shape, baseline.shape)

    # plt.plot(pred[:, 0])
    # plt.plot(baseline[:, 0])
    # plt.plot(target[:, 0])
    # plt.show()

    loss_pred = torch.sqrt(torch.mean((pred-target)**2, dim=0))
    loss_base = torch.sqrt(torch.mean((baseline-target)**2, dim=0))

    # loss_pred = np.array(loss_pred)
    # loss_base = np.array(loss_base)
    print("the loss of pred on AVAL: {:.4f}".format(loss_pred[0]))
    print("the loss of baseline on AVAL: {:.4f}".format(loss_base[0]))
    print("-------")
    print("the loss of pred on 302 neurons: {:.4f}".format(loss_pred.mean()))
    print("the loss of baseline on 302 neurons: {:.4f}".format(loss_base.mean()))
    print("-------")


    plt.scatter(labels_neurons_with_data, loss_pred)
    plt.scatter(labels_neurons_with_data, loss_base)
    plt.legend(["pred", "base"])
    plt.xlabel("Neuron")
    plt.ylabel("Loss(MSE)")
    plt.show()

    print("the closest: {:.4f}".format(float((loss_pred-loss_base).min())))
    print(float(loss_pred[1] - loss_base[1]))
    idx = torch.argmin((loss_pred - loss_base), dim=0)


    plt.scatter(labels_neurons_with_data, loss_pred-loss_base, edgecolors='g')
    plt.legend(["pred - base"])
    plt.xlabel("Neuron")
    plt.ylabel("Loss(MSE)")
    plt.show()