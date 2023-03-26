#!/usr/bin/env python
# encoding: utf-8
import math

from govfunc._utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, len, epsilon):
        super(Net, self).__init__()
        self.seq_len = len
        self.tau = epsilon
        self.alpha = torch.nn.Parameter(torch.zeros(1, 1))

    def get_exp(self):
        matrix = [math.e ** (-k * 1.0 / self.tau) for k in range(1, seq_len + 1)]
        sum_result = np.array(matrix).sum()
        coef = [math.e ** (-j * 1.0 / self.tau) / sum_result for j in range(1, seq_len + 1)]
        coef = torch.tensor(coef).to(torch.float32)
        return coef

    def forward(self, input):
        coef = self.get_exp()
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
    # tau_range = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1, 2, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60]
    tau_range = [10]
    # different choices for seq_len
    seq_len_range = [10, 20, 50, 100, 500, 1000, 3000]
    epoch = 100

    list_save = []
    for no in neuron_range:

        alpha_list = []
        for seq_len in seq_len_range:
            residual = single_worm_dataset["residual_calcium"][:, no].to(torch.float32)

            input_group = []
            target_group = []
            # prepare batches of samples
            for j in range(seq_len, residual.shape[0]):
                cal_input = residual[j - seq_len: j].T
                target = residual[j:j + 1].T.squeeze()
                input_group.append(np.array(cal_input))
                target_group.append(np.array(target))

            train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(input_group)),
                                                           torch.tensor(np.array(target_group)))
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=128,
                shuffle=True,
            )

            for tau in tau_range:
                # initialize network hyperparameters
                net = Net(seq_len, tau)
                optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
                loss_func = torch.nn.MSELoss()
                residual = residual.to(torch.float32)

                for e in range(epoch):
                    for cal_input, target in train_loader:
                        prediction = net.forward(cal_input).squeeze()
                        loss = loss_func(prediction, target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print("batch, ", net.state_dict()["alpha"].item())
                # save the alpha that model finded
                alpha_list.append(net.state_dict()["alpha"].item())
            # for each neuron: for each tau model had an optimal alpha
        list_save.append(alpha_list)

    for k in range(len(list_save)):
        if k == len(list_save) - 1:
            plt.plot(seq_len_range, list_save[k], linewidth=3)
        else:
            plt.plot(seq_len_range, list_save[k])
    list_name = neuron_range
    list_name.pop()
    list_name.append("all neurons")
    plt.legend(list_name)
    plt.title("best alpha --- seq_len curve (tau = 10)")
    plt.xlabel("seq_len")
    plt.ylabel("optimal solution of alpha")
    plt.show()
