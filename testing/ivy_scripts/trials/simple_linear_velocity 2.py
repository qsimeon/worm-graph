#!/usr/bin/env python
# encoding: utf-8
import math

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


class Net(nn.Module):
    def __init__(self, in_, out_):
        super(Net, self).__init__()
        self.predict = torch.nn.Linear(in_, out_, bias=False)

    def forward(self, input):
        out = self.predict(input)
        return out


def alpha_relation(velocity, residual, alpha, seq_len, tau):
    train_border = 1000
    matrix = [math.e ** (-i) for i in range(1, seq_len + 1)]
    sum_result = np.array(matrix).sum()
    coef = [math.e ** (-i) * alpha / sum_result for i in range(1, seq_len + 1)]
    coef = torch.tensor(coef).T.to(torch.float32)
    val_loss_history = []
    loss_func = torch.nn.MSELoss()
    for t in range(seq_len, train_border - tau):
        input = residual[t - seq_len: t].T
        target = residual[t:t + tau].T
        prediction = input @ coef
        # print(coef.shape, input.shape, target.shape, prediction.shape)
        loss = loss_func(prediction, target)
        val_loss_history.append(loss.detach().numpy())
    val_loss_history = np.array(val_loss_history)
    # print("mean val loss = ", val_loss_history.mean())
    return val_loss_history.mean()


def seq_diff(velocity, residual, seq_len, tau, epoch):
    net = Net(seq_len, 1)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    train_border = 3000
    for e in range(epoch):
        for t in range(seq_len, train_border - seq_len):
            input = velocity[t - seq_len: t].T
            target = residual[t: t + tau].T
            prediction = net.forward(input)
            loss = loss_func(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(net.state_dict()["predict.weight"])

    w1 = net.state_dict()["predict.weight"][0][0]
    # w2 = net.state_dict()["predict.weight"][0][1]
    print("alpha = ", w1)

    # on validation set
    net.eval()
    val_loss_history = []
    for t in range(train_border, residual.shape[0] - tau - seq_len):
        input = velocity[t: t + seq_len].T
        target = residual[t + seq_len + 1: t + seq_len + 1 + tau].T
        prediction = net.forward(input)
        loss = loss_func(prediction, target)
        val_loss_history.append(loss.detach().numpy())
    val_loss_history = np.array(val_loss_history)
    print("min val loss = ", val_loss_history.min())


if __name__ == "__main__":
    dataset = load_Uzel2022()

    # here we just use one worm
    single_worm_dataset = dataset["worm0"]

    name_mask = single_worm_dataset["named_neurons_mask"]

    dict_setting = {0: 12, 1: 22, 2: name_mask}

    for i in range(len(l)):
        calcium_data = single_worm_dataset["calcium_data"][:, dict_setting[i]].to(torch.float32)
        residual = single_worm_dataset["residual_calcium"][:, dict_setting[i]].to(torch.float32)

        dx_nan = torch.div(residual, single_worm_dataset["dt"])
        velocity = torch.where(torch.isnan(dx_nan), torch.full_like(dx_nan, 0), dx_nan).to(torch.float32)
        cols_with_data_mask = name_mask
        labels_neurons_with_data = [
            single_worm_dataset["slot_to_neuron"][slot]
            for slot, boole in enumerate(cols_with_data_mask)
            if boole.item() is True
        ]


        for seq_len in range(1, 100, 10):
            tau = 1
            alpha_range = []
            mean_val_loss = []
            for alpha in range(-100, 100):
                alpha *= 0.01
                alpha_range.append(alpha)
                loss = alpha_relation(velocity, residual, alpha, seq_len, tau)
                mean_val_loss.append(loss)

            print(alpha_range[mean_val_loss.index(np.array(mean_val_loss).min())])
            plt.plot(alpha_range, mean_val_loss)
            plt.ylabel("mean val loss")
            plt.xlabel("alpha")
            plt.title("seq_len = " + str(seq_len) + ", tau = 1")
            plt.show()

    # pred = w2 * cal_data[3000:-1] + w1 * cal_data[2999:-2]
    # baseline = cal_data[3000:-1]
    # target = cal_data[3001:]
    # loss_pred = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
    # loss_base = torch.sqrt(torch.mean((baseline - target) ** 2, dim=0))

    # print("the loss of pred on AVAL: {:.4f}".format(loss_pred[0]))
    # print("the loss of baseline on AVAL: {:.4f}".format(loss_base[0]))
    # print("-------")
    # print("the loss of pred on 302 neurons: {:.4f}".format(loss_pred.mean()))
    # print("the loss of baseline on 302 neurons: {:.4f}".format(loss_base.mean()))

    # plt.scatter(labels_neurons_with_data, loss_pred)
    # plt.scatter(labels_neurons_with_data, loss_base)
    # plt.legend(["pred", "base"])
    # plt.xlabel("Neuron")
    # plt.ylabel("Loss(MSE)")
    # plt.show()
    #
    # print("the closest: {:.4f}".format(float((loss_pred - loss_base).min())))
    #
    # plt.scatter(labels_neurons_with_data, loss_pred - loss_base, edgecolors="g")
    # plt.legend(["pred - base"])
    # plt.xlabel("Neuron")
    # plt.ylabel("Loss(MSE)")
    # plt.show()

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
    data = pd.DataFrame.from_dict(
        {"x": labels_neurons_with_data, "base": loss_base, "pred": loss_pred}
    )

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="base", y="x", data=data, label="base", color="b")

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="pred", y="x", data=data, label="pred", color="b")
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
