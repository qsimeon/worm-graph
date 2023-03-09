#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: smooth_method_trial.py
@time: 2023/3/3 16:36
"""

### smooth method: TVR, Savitzky-Golay filter, np.convolve()


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


def derivative(y, t):
    """
    input: [time, status]
    func: calculate the residual between time steps
    output: [residual(\delta t), status]
    """
    yrow, ycol = y.size()
    dy = np.zeros((yrow - 1, ycol))
    for i in range(0, yrow - 1):
        dy[i, :] = y[i + 1, :] - y[i, :]
    return dy


if __name__ == "__main__":
    # # Data
    # dx = 0.01
    #
    # data = []
    # for x in np.arange(0, 1, dx):
    #     data.append(abs(x - 0.5))
    # data = np.array(data)
    #
    # # True derivative
    # deriv_true = []
    # for x in np.arange(0, 1, dx):
    #     if x < 0.5:
    #         deriv_true.append(-1)
    #     else:
    #         deriv_true.append(1)
    # deriv_true = np.array(deriv_true)
    #
    # # Add noise
    # n = len(data)
    # data_noisy = data + np.random.normal(0, 0.05, n)

    config = OmegaConf.load("dataset.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)
    print("----dataset prepared------\n")

    # proof on data preprocessing
    print(dataset["worm0"].keys())
    print(dataset["worm0"]["calcium_data"].shape, dataset["worm0"]["residual_calcium"].shape)
    ### 'calcium_data', 'smooth_calcium_data', 'residual_calcium', 'residual_smooth_calcium',

    # n = dataset["worm0"]["calcium_data"].shape[0]
    # t = dataset["worm0"]["calcium_data"][1:] - dataset["worm0"]["calcium_data"][:(n-1)]
    # print("zzzz", t.shape)
    print(dataset["worm0"]["calcium_data"][1:] - dataset["worm0"]["calcium_data"][:-1] == dataset["worm0"][
                                                                                              "residual_calcium"][1:])

    plt.plot(dataset["worm0"]["calcium_data"][:, 0])
    plt.plot(dataset["worm0"]["smooth_calcium_data"][:, 0])
    plt.legend(["cal", "smooth_cal"])
    plt.show()

    exit(0)

    ########### neuronal trials ##################
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

    print(worm[0].keys())
    # dict_keys(['dataset', 'worm', 'calcium_data', 'smooth_calcium_data', 'residual_calcium', 'residual_smooth_calcium', 'neuron_to_idx', 'idx_to_neuron', 'max_time', 'num_neurons', 'num_named_neurons', 'num_unknown_neurons', 'named_neurons_mask', 'unknown_neurons_mask', 'neurons_mask', 'named_neuron_to_idx', 'idx_to_named_neuron', 'unknown_neuron_to_idx', 'idx_to_unknown_neuron', 'slot_to_named_neuron', 'named_neuron_to_slot', 'slot_to_unknown_neuron', 'unknown_neuron_to_slot', 'slot_to_neuron', 'neuron_to_slot'])
    exit(0)

    print(data_torch.shape)

    print("---")
    print(data_torch.shape)

    # FFT
    filtered_data_torch = torch.zeros_like(data_torch)
    max_time, num_neurons = data_torch.shape
    frequencies = torch.fft.rfftfreq(max_time, d=1.0)
    threshold = torch.abs(frequencies)[30]  # picks first 30 frequencies (can use value > 30 to smooth less)
    oneD_kernel = torch.abs(frequencies) < threshold
    fft_input = torch.fft.rfftn(data_torch, dim=0)
    print(fft_input.shape, oneD_kernel.shape)
    oneD_kernel = oneD_kernel.repeat(302, 1).T
    filtered_data_torch[0:] = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
    print(filtered_data_torch.shape, "-----------")

    plt.plot(data_torch[:, 26])
    plt.plot(filtered_data_torch[:, 26])
    plt.title("calcium data")
    plt.legend(["True", "FFT"])
    # fig2.savefig('derivative.png')
    plt.show()

    # plt.semilogy(fft_input)
    # plt.semilogy(fft_input * oneD_kernel)
    # plt.semilogy(oneD_kernel)  # frequencies are in Hertz (if we knew the real `dt`)
    # plt.xlabel("Hz")
    # plt.ylabel("Amplitude")
    # plt.title("trial")
    # plt.grid()
    # plt.show()
    # exit(0)
