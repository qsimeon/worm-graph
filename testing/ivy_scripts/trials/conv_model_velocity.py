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



    list_save = []
    for no in [12, 22, 59, name_mask]:
        residual = single_worm_dataset["residual_calcium"][:, no].to(torch.float32)

        seq_len = 3000
        tau_range = [0.01, 0.1, 1, 10, 100]
        alpha_list = []
        epoch = 100

        input_group = []
        target_group = []
        for j in range(seq_len, residual.shape[0]):
            input = residual[j - seq_len: j].T
            target = residual[j:j + 1].T.squeeze()
            input_group.append(np.array(input))
            target_group.append(np.array(target))

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(input_group)),
                                                       torch.tensor(np.array(target_group)))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=128,
            shuffle=True,
        )

        for tau in tau_range:
            net = Net(seq_len, tau)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()
            residual = residual.to(torch.float32)

            for e in range(epoch):
                for input, target in train_loader:
                    prediction = net.forward(input).squeeze()
                    loss = loss_func(prediction, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("batch, long, ", net.state_dict()["alpha"].item())
            alpha_list.append(net.state_dict()["alpha"].item())

    plt.plot(range(len(tau_range)), alpha_list)
    plt.title("neuron " + str(no))
    plt.show()
