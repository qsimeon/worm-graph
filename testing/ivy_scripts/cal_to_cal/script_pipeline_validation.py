#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: script_pipeline_validation.py
@time: 2023/3/10 14:54
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from train._main import *

import numpy as np
import matplotlib.pyplot as plt
import torch.fft as fft
import torch

from torch.autograd import Variable
from models._main import *
from preprocess._utils import *
import torch.nn as nn
import torch.utils.data as Data
import torch
from torch.optim import optimizer


def create_synthetic_data(d, n, ifnoise=False):
    res = np.zeros((d, n))
    assert isinstance(n, int), "wrong number for samples"
    der = []
    for i in range(0, n):
        freq = np.random.uniform(1.0 / d, 5 * 1.0 / d)
        phi = np.random.random()
        step = np.arange(d)
        res[:, i] = np.sin(2 * np.pi * freq * step + phi * (np.pi / 180))
        der.append(2 * np.pi * freq)

    return res, der


def create_dataset(raw_data, raw_der):
    sine_dataset = dict()
    for i, real_data in enumerate(raw_data):
        worm = "worm" + str(i)
        max_time = seq_len
        num_neurons = num_signal
        der = np.array(raw_der[i])
        der = der.reshape(der.shape[0], 1).T
        time_in_seconds = torch.tensor(np.array(np.arange(seq_len)).reshape(seq_len, 1))
        dt = torch.tensor(der)

        num_named = num_neurons
        real_data = torch.tensor(real_data, dtype=torch.float64)

        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data, "fft"
        )

        for i in range(residual.shape[1]):
            residual[:, i] = residual[:, i] / dt[:, i]

        sine_dataset.update(
            {
                worm: {
                    "dataset": "sine",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": range(0, num_neurons),
                    "idx_to_neuron": range(num_neurons - 1, -1, -1),
                    "max_time": int(max_time),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "named_neurons_mask": torch.full((num_neurons,), True),
                    "named_neuron_to_idx": range(0, num_neurons),
                    "idx_to_named_neuron": range(num_neurons - 1, -1, -1),
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
    return sine_dataset


class lstm(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, LAYERS, DROP_RATE):
        super(lstm, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True,
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, INPUT_SIZE)
        self.h_s = None
        self.h_c = None

    def forward(self, x):
        r_out, (h_s, h_c) = self.rnn(x)
        output = self.hidden_out(r_out)
        return output


if __name__ == "__main__":
    # Creating signal
    seq_len = 3312
    num_signal = 302
    if_noise = False
    num_worms = 6
    raw_data = []
    raw_der = []
    for j in range(num_worms):
        x, der = create_synthetic_data(seq_len, num_signal, if_noise)
        x_torch = Variable(torch.from_numpy(x), requires_grad=False)
        raw_data.append(x_torch)
        raw_der.append(der)

    dataset = create_dataset(raw_data, raw_der)
    print(dataset["worm0"].keys())

    plt.plot(dataset["worm0"]["calcium_data"][:, 3])
    plt.plot(dataset["worm0"]["residual_calcium"][:, 3])
    plt.legend(["cal", "res"], loc="upper right")
    plt.show()

    config = OmegaConf.load("../../../../conf/model.yaml")
    print("Model:", OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(config)

    config = OmegaConf.load("../../../../conf/train.yaml")
    model, log_dir = train_model(model, dataset, config)
