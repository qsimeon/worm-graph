#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: create_sine_dataset.py
@time: 2023/3/14 10:01
'''

# !/usr/bin/env python
# encoding: utf-8
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

def smooth_data(calcium_data, smooth_method, dt=1.0):
    """
    Smooth the calcium data. Returns the denoised signals calcium signals
    using FFT.

    Args:
        calcium_data: original calcium data from dataset
        smooth_method: the way to smooth data
        dt: (required when use FFT as smooth_method) the sampling time (unit: sec)

    Returns:
        smooth_ca_data: calcium data that are smoothed
        residual: original residual (calculated by calcium_data)
        residual_smooth_ca_data: residual calculated by smoothed calcium data
    """
    n = calcium_data.shape[0]
    # initialize the size for smooth_calcium_data
    smooth_ca_data = torch.zeros_like(calcium_data)
    # calculate original residual
    # residual = torch.zeros_like(calcium_data)
    # residual[1:] = calcium_data[1:] - calcium_data[: n - 1]
    if str(smooth_method).lower() == "sg" or smooth_method == None:
        smooth_ca_data = savgol_filter(calcium_data, 5, 3, mode="nearest", axis=-1)
    elif str(smooth_method).lower() == "fft":
        data_torch = calcium_data
        smooth_ca_data = torch.zeros_like(calcium_data)
        max_time, num_neurons = data_torch.shape
        frequencies = torch.fft.rfftfreq(max_time, d=dt)  # dt: sampling time
        threshold = torch.abs(frequencies)[int(frequencies.shape[0] * 0.1)]
        oneD_kernel = torch.abs(frequencies) < threshold
        fft_input = torch.fft.rfftn(data_torch, dim=0)
        oneD_kernel = oneD_kernel.repeat(calcium_data.shape[1], 1).T
        fft_result = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
        smooth_ca_data[0 : min(fft_result.shape[0], calcium_data.shape[0])] = fft_result
    elif str(smooth_method).lower() == "tvr":
        diff_tvr = DiffTVR(n, 1)
        for i in range(0, calcium_data.shape[1]):
            temp = np.array(calcium_data[:, i])
            temp.reshape(len(temp), 1)
            (item_denoise, _) = diff_tvr.get_deriv_tvr(
                data=temp,
                deriv_guess=np.full(n + 1, 0.0),
                alpha=0.005,
                no_opt_steps=100,
            )
            smooth_ca_data[:, i] = torch.tensor(item_denoise[: (len(item_denoise) - 1)])
    else:
        print("Wrong Input, check the config/preprocess.yml")
        exit(0)
    m = smooth_ca_data.shape[0]
    # residual_smooth_ca_data = torch.zeros_like(residual)
    # residual_smooth_ca_data[1:] = smooth_ca_data[1:] - smooth_ca_data[: m - 1]
    return smooth_ca_data # residual, residual_smooth_ca_data


def create_synthetic_data(d, n, ifnoise=False, sum=0):
    calcium = np.zeros((d, n))
    res = np.zeros((d, n))

    assert isinstance(n, int), "wrong number for samples"
    der = []
    for i in range(0, n):
        freq = 0
        step = np.arange(d)
        freq1 = np.random.uniform(1.0 / d, 5 * 1.0 / d)
        freq += freq1
        phi1 = np.random.random()
        calcium[:, i] = np.sin(2 * np.pi * freq1 * step + phi1 * (np.pi / 180))
        for k in range(random.randint(0, sum)):
            freq2 = np.random.uniform(1.0 / d, 5 * 1.0 / d)
            phi2 = np.random.random()
            calcium[:, i] += np.sin(2 * np.pi * freq2 * step + phi2 * (np.pi / 180))
            freq += freq2
        if ifnoise:
            for j in range(0, d):
                calcium[j, i] += + random.gauss(0, 0.02)

        res[1:, i] = (calcium[1:, i] - calcium[:-1, i])

    return calcium, res


def create_dataset(raw_data, raw_res):
    sine_dataset = dict()
    for i, real_data in enumerate(raw_data):
        worm = "worm" + str(i)
        max_time = seq_len
        num_neurons = num_signal
        time_in_seconds = torch.tensor(np.array(np.arange(seq_len)).reshape(seq_len, 1))

        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )
        residual = torch.tensor(
            raw_res[i], dtype=torch.float64
        )


        smooth_real_data = smooth_data(real_data, "fft")

        residual_smooth_calcium = torch.zeros_like(smooth_real_data)
        residual_smooth_calcium[1:, :] = smooth_real_data[1:, :] - smooth_real_data[:-1, :]

        # randomly choose some neurons to be inactivated
        num_unnamed = 100
        list_random = random.sample(range(1, real_data.shape[1]), num_unnamed)
        named_mask = torch.full((num_neurons,), True)

        for i in list_random:
            real_data[:, i] = torch.zeros_like(real_data[:, 0])
            residual[:, i] = torch.zeros_like(residual[:, 0])
            named_mask[i] = False

        dt = torch.ones(real_data.shape[0], 1)

        sine_dataset.update(
            {
                worm: {
                    "dataset": "sine",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": residual_smooth_calcium,
                    "neuron_to_idx": range(0, num_neurons),
                    "idx_to_neuron": range(num_neurons - 1, -1, -1),
                    "max_time": int(max_time),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "named_neurons_mask": named_mask,
                    "named_neuron_to_idx": list_random,
                    "idx_to_named_neuron": list_random,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": int(num_neurons) - num_unnamed,
                    "num_unknown_neurons": num_unnamed,
                },
            }
        )
    return sine_dataset


# Creating signal
seq_len = 3312
num_signal = 302
if_noise = False
sum = 8
num_worms = 6
raw_data = []
raw_der = []
for j in range(num_worms):
    x, dx = create_synthetic_data(seq_len, num_signal, if_noise, sum)
    x_torch = Variable(torch.from_numpy(x), requires_grad=False)
    raw_data.append(x_torch)
    raw_der.append(dx)

dataset = create_dataset(raw_data, raw_der)

print(dataset["worm0"].keys())
# print(dataset["worm0"]["dt"])

plt.plot(dataset["worm0"]["calcium_data"][:, 0:10])
# plt.plot(dataset["worm0"]["residual_calcium"][:, 0:10])
plt.show()

plt.plot(dataset["worm0"]["calcium_data"][:, 0])

plt.plot(dataset["worm0"]["residual_calcium"][:, 0])
plt.legend(["cal", "res"])
plt.show()


plt.plot(dataset["worm0"]["smooth_calcium_data"][:, 0])

plt.plot(dataset["worm0"]["residual_smooth_calcium"][:, 0])

plt.legend(["cal", "res"])
plt.show()

print(dataset["worm0"]["named_neurons_mask"])

file = open("./data/processed/neural/sum_sine.pickle", "wb")
pickle.dump(dataset, file)

file.close()
