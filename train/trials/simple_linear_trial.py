#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: smooth_method_trial.py
@time: 2023/3/3 16:36
"""
import pandas

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
import torch.nn.functional as F

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

    print(worm[0]["time_in_seconds"].shape, worm[0]["dt"].shape)

    temp = torch.stack((worm[0]["calcium_data"], worm[0]["residual_calcium"]), 2)
    tmp = temp[:, :, 0]
    print(tmp.shape)
    plt.plot(tmp[:, 2])
    plt.show()

    print(temp.shape)
    exit(0)

    # print(worm[0].keys())
    # print(worm[0]["neuron_to_idx"],'\n', worm[0]["num_neurons"], worm[0]["num_named_neurons"], worm[0]["num_unknown_neurons"])

    # try the basic model
    cols_with_data_mask = worm[0]["neurons_mask"]
    labels_neurons_with_data = [worm[0]["slot_to_neuron"][slot] for slot, boole in enumerate(cols_with_data_mask) if
                                boole.item() is True]

    cal_data = worm[0]["smooth_calcium_data"][:, cols_with_data_mask]
    data_temp = cal_data

    # # how smooth
    # plt.plot(cal_data[:, 0])
    # plt.title("smooth visualization")
    # plt.show()

    cal_data = cal_data[0:3000]


    class Net(nn.Module):
        def __init__(self, in_, out_):
            super(Net, self).__init__()
            self.predict = torch.nn.Linear(in_, out_)

        def forward(self, input):
            out = self.predict(input)
            return out


    net = Net(2, 1)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    epoch = 1000
    cal_data = cal_data.to(torch.float32)
    for e in range(epoch):
        for t in range(1, cal_data.shape[0] - 1):
            input = cal_data[t - 1:t + 1].T
            target = cal_data[t + 1:t + 2].T
            prediction = net.forward(input)
            loss = loss_func(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # for name in net.state_dict():
    #     print(name)

    print(net.state_dict()["predict.weight"])

    w1 = net.state_dict()["predict.weight"][0][0]
    w2 = net.state_dict()["predict.weight"][0][1]
    print(w1, w2)

    cal_data = data_temp
    pred = w2 * cal_data[3000:-1] + w1 * cal_data[2999:-2]
    baseline = cal_data[3000:-1]
    target = cal_data[3001:]
    loss_pred = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
    loss_base = torch.sqrt(torch.mean((baseline - target) ** 2, dim=0))

    print("the loss of pred on AVAL: {:.4f}".format(loss_pred[0]))
    print("the loss of baseline on AVAL: {:.4f}".format(loss_base[0]))
    print("-------")
    print("the loss of pred on 302 neurons: {:.4f}".format(loss_pred.mean()))
    print("the loss of baseline on 302 neurons: {:.4f}".format(loss_base.mean()))

    plt.scatter(labels_neurons_with_data, loss_pred)
    plt.scatter(labels_neurons_with_data, loss_base)
    plt.legend(["pred", "base"])
    plt.xlabel("Neuron")
    plt.ylabel("Loss(MSE)")
    plt.show()

    print("the closest: {:.4f}".format(float((loss_pred - loss_base).min())))

    plt.scatter(labels_neurons_with_data, loss_pred - loss_base, edgecolors='g')
    plt.legend(["pred - base"])
    plt.xlabel("Neuron")
    plt.ylabel("Loss(MSE)")
    plt.show()

    exit(0)








    ###################### for the simplest linear combination on neuronal signals #################
    # pred y_t+1 = wa * y_t + wb * y_t-1
    wa = 1.5
    wb = -0.5
    cal_data = data_temp
    pred = wa * cal_data[3000:-1] + wb * cal_data[2999:-2]
    # baseline: y_t+1 == y_t
    baseline = cal_data[3000:-1]
    # target: the real data for y_t+1
    target = cal_data[3001:]
    print(target.shape, pred.shape, baseline.shape)

    loss_pred = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
    loss_base = torch.sqrt(torch.mean((baseline - target) ** 2, dim=0))

    # loss_pred = np.array(loss_pred)
    # loss_base = np.array(loss_base)
    print("the loss of pred on AVAL: {:.4f}".format(loss_pred[0]))
    print("the loss of baseline on AVAL: {:.4f}".format(loss_base[0]))
    print("-------")
    print("the loss of pred on 302 neurons: {:.4f}".format(loss_pred.mean()))
    print("the loss of baseline on 302 neurons: {:.4f}".format(loss_base.mean()))
    print("-------")

    # zipped = zip(labels_neurons_with_data, loss_base, loss_pred)
    # zipped = sorted(zipped)
    #
    # labels_neurons_with_data, loss_base, loss_pred = (list(t) for t in zip(*zipped))

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create the dataframe with your data
    data = pd.DataFrame.from_dict({"x": labels_neurons_with_data, "base": loss_base, "pred": loss_pred})

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="base", y="x", data=data,
                label="base", color="b")

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="pred", y="x", data=data,
                label="pred", color="b")
    plt.show()

    # sns.set_theme(style="whitegrid")
    #
    # # Initialize the matplotlib figure
    # f, ax = plt.subplots(figsize=(6, 15))
    #
    # # Load the example car crash dataset
    # crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)
    #
    # # Plot the total crashes
    # sns.set_color_codes("pastel")
    # sns.barplot(x="base", y="abbrev", data=loss_base,
    #             label="Base", color="b")
    #
    # # Plot the crashes where alcohol was involved
    # sns.set_color_codes("muted")
    # sns.barplot(x="alcohol", y="abbrev", data=loss_pred,
    #             label="Pred", color="b")
    #
    # # Add a legend and informative axis label
    # ax.legend(ncol=2, loc="lower right", frameon=True)
    # ax.set(xlim=(0, 1), ylabel="",
    #        xlabel="Loss (MSE)")
    # sns.despine(left=True, bottom=True)
