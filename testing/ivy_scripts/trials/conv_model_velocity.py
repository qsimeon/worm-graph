#!/usr/bin/env python
# encoding: utf-8
import math

from govfunc._utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, tau):
        super(Net, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1, 1))
        self.tau = tau

    def weight_exp(self, input):
        # input shape: [neurons, seq_len] or [batch_size, neurons, seq_len]
        seq_len = input.shape[1]
        matrix = np.array([math.e ** (-k * 1.0 / self.tau) for k in range(1, seq_len + 1)])
        sum_result = matrix.sum()
        coef = [math.e ** (-k * 1.0 / self.tau) / sum_result for k in range(1, seq_len + 1)]
        coef = torch.tensor(coef).to(torch.float32)
        return coef

    def forward(self, input):
        coef = self.weight_exp(input)
        weighted_history = input @ coef
        res_t = self.alpha * weighted_history
        return res_t


if __name__ == "__main__":
    dataset = load_Uzel2022()
    # here we just use one worm
    single_worm_dataset = dataset["worm0"]
    name_mask = single_worm_dataset["named_neurons_mask"]

    neuron_range = [12, 22, 59, name_mask]

    # different choices for tau
    half_life_seq_len_range = range(10, 500, 90)
    # # different choices for seq_len
    # seq_len_range = [1]
    epoch = 100

    list_save = []
    for no in neuron_range:
        if no is not name_mask:
            residual = single_worm_dataset["residual_calcium"][:, no].reshape(
                single_worm_dataset["residual_calcium"].shape[0], 1).to(
                torch.float32)
        else:
            residual = single_worm_dataset["residual_calcium"][:, no].to(
                torch.float32)

        input_group = []
        target_group = []
        for t in range(1, residual.shape[0] - 2):
            cal_input = residual[0:t].T
            target = residual[t:t + 1].T.squeeze()
            input_group.append(cal_input)
            target_group.append(target)
        # train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(input_group)),
        #                                                torch.tensor(np.array(target_group)))
        # train_loader = torch.utils.data.DataLoader(
        #     dataset=train_dataset,
        #     batch_size=128,
        #     shuffle=True,
        # )

        alpha_list = []
        # prepare batches of samples
        for half_seq_len in half_life_seq_len_range:
            # initialize network hyperparameters
            net = Net(half_seq_len)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()

            for e in range(epoch):
                for i in range(len(input_group)):
                    cal_input = input_group[i]
                    target = target_group[i]
                    prediction = net.forward(cal_input).squeeze()
                    loss = loss_func(prediction, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("half_seq_len = ", half_seq_len, "alpha = ", net.state_dict()["alpha"].item())
            alpha_list.append(net.state_dict()["alpha"].item())
        # for each neuron: for each tau model had an optimal alpha
        list_save.append(alpha_list)

    for k in range(len(list_save)):
        if k == len(list_save) - 1:
            plt.plot(half_life_seq_len_range, list_save[k], linewidth=2)
        else:
            plt.plot(half_life_seq_len_range, list_save[k])

    print(list_save[-1][-1])
    list_name = neuron_range
    list_name.pop()
    list_name.append("all neurons")
    plt.legend(list_name)
    plt.title("alpha - half_life_seq_len_range")
    plt.xlabel("half_life_seq_len_range, epoch = 10")
    plt.ylabel("optimal solution of alpha")
    plt.show()
