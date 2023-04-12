#!/usr/bin/env python
# encoding: utf-8
import math

from govfunc._utils import *
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
        input = residual[t - seq_len : t].T
        target = residual[t : t + tau].T
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
            input = velocity[t - seq_len : t].T
            target = residual[t : t + tau].T
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
        input = velocity[t : t + seq_len].T
        target = residual[t + seq_len + 1 : t + seq_len + 1 + tau].T
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

    seq_range = range(1, 100, 3)

    dict_setting = {0: 12, 1: 22, 2: 59, 3: name_mask}

    seq_for_neurons = []
    for i in range(len(dict_setting)):
        calcium_data = single_worm_dataset["calcium_data"][:, dict_setting[i]].to(
            torch.float32
        )
        residual = single_worm_dataset["residual_calcium"][:, dict_setting[i]].to(
            torch.float32
        )

        dx_nan = torch.div(residual, single_worm_dataset["dt"])
        velocity = torch.where(
            torch.isnan(dx_nan), torch.full_like(dx_nan, 0), dx_nan
        ).to(torch.float32)
        cols_with_data_mask = name_mask
        labels_neurons_with_data = [
            single_worm_dataset["slot_to_neuron"][slot]
            for slot, boole in enumerate(cols_with_data_mask)
            if boole.item() is True
        ]

        alpha_for_one_seq = []
        for seq_len in seq_range:
            tau = 1
            alpha_range = []
            mean_val_loss = []
            for alpha in range(-1000, 1000, 5):
                alpha *= 0.001
                alpha_range.append(alpha)
                loss = alpha_relation(velocity, residual, alpha, seq_len, tau)
                mean_val_loss.append(loss)
            alpha_for_one_seq.append(
                alpha_range[mean_val_loss.index(np.array(mean_val_loss).min())]
            )
        seq_for_neurons.append(alpha_for_one_seq)

    plt.xlabel("seq_len")
    plt.ylabel("best solution for alpha")
    list_neuron_name = []
    for k in range(len(dict_setting)):
        if k is not len(dict_setting) - 1:
            plt.plot(seq_range, seq_for_neurons[k])
            list_neuron_name.append(
                single_worm_dataset["slot_to_neuron"][dict_setting[k]]
            )
        else:
            plt.plot(seq_range, seq_for_neurons[k])
            list_neuron_name.append("all named neurons")
    plt.legend(list_neuron_name, loc="upper right")
    plt.show()
