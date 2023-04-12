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
        # input shape: [batch_size, neurons, seq_len]
        seq_len = input.shape[2]
        matrix = np.array(
            [math.e ** (-p * 1.0 / self.tau) for p in range(1, seq_len + 1)]
        )
        sum_result = matrix.sum()
        coef = [
            math.e ** (-q * 1.0 / self.tau) / sum_result for q in range(1, seq_len + 1)
        ]
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
    # half_life_seq_len_range = [1]
    # # different choices for seq_len
    seq_len_range = range(1, 100, 2)
    epoch = 1

    list_save = []
    for no in neuron_range:
        if no is not name_mask:
            residual = (
                single_worm_dataset["residual_calcium"][:, no]
                .reshape(single_worm_dataset["residual_calcium"].shape[0], 1)
                .to(torch.float32)
            )
        else:
            residual = single_worm_dataset["residual_calcium"][:, no].to(torch.float32)

        alpha_list = []
        for seq_len in seq_len_range:
            # initialize network hyperparameters
            net = Net(1)
            optimizer = torch.optim.SGD(net.parameters(), lr=1)
            loss_func = torch.nn.MSELoss()

            input_group = []
            target_group = []
            for t in range(seq_len, residual.shape[0] - 1):
                cal_input = residual[t - seq_len : t].T
                target = residual[t : t + 1].T.squeeze()
                input_group.append(np.array(cal_input))
                target_group.append(np.array(target))
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(np.array(input_group)),
                torch.tensor(np.array(target_group)),
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=128,
                shuffle=True,
            )

            # prepare batches of samples
            for e in range(epoch):
                for cal_input, target in train_loader:
                    prediction = net.forward(cal_input).squeeze()
                    loss = loss_func(prediction, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("seq_len = ", seq_len, "alpha = ", net.state_dict()["alpha"].item())
            alpha_list.append(net.state_dict()["alpha"].item())
        # for each neuron: for each tau model had an optimal alpha
        list_save.append(alpha_list)

    for k in range(len(list_save)):
        if k == len(list_save) - 1:
            plt.plot(seq_len_range, list_save[k], linewidth=2)
        else:
            plt.plot(seq_len_range, list_save[k])

    print(list_save[-1][-1])

    neuron_range.pop()
    list_name = [single_worm_dataset["slot_to_neuron"][i] for i in neuron_range]
    list_name.append("all neurons")
    plt.legend(list_name)
    plt.title("alpha - seq_len_range")
    plt.xlabel("seq_len_range, epoch = 1, tau = 1")
    plt.ylabel("optimal solution of alpha")
    plt.show()
