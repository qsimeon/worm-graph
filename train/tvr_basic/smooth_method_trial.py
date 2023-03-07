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

    print(data_torch.shape)

    print("---")
    print(data_torch.shape)

    max_time, num_neurons = data_torch.shape
    frequencies = torch.fft.rfftfreq(max_time, d=1.0)
    threshold = torch.abs(frequencies)[30]  # picks first 30 frequencies (can use value > 30 to smooth less)
    oneD_kernel = torch.abs(frequencies) < threshold
    fft_input = torch.fft.rfftn(data_torch, dim=0)
    oneD_kernel = oneD_kernel.repeat(302, 1).T
    filtered_data_torch = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
    plt.semilogy(fft_input)
    plt.semilogy(fft_input * oneD_kernel)
    plt.semilogy(oneD_kernel)  # frequencies are in Hertz (if we knew the real `dt`)
    plt.xlabel("Hz")
    plt.ylabel("Amplitude")
    plt.title("trial")
    plt.grid()
    plt.show()
    exit(0)

    n = data_torch.shape[0]
    frequencies = torch.fft.rfftfreq(n, d=1.0)
    threshold = torch.abs(frequencies)[30]
    oneD_kernel = torch.abs(frequencies) < threshold
    fft_input = torch.fft.rfft(data_torch)
    new_yf_clean = torch.fft.irfft(fft_input * oneD_kernel)


    plt.semilogy(fft_input)
    plt.semilogy(fft_input * oneD_kernel)
    plt.semilogy(oneD_kernel)  # frequencies are in Hertz (if we knew the real `dt`)
    plt.xlabel("Hz")
    plt.ylabel("Amplitude")
    plt.title("trial")
    plt.grid()
    plt.show()
    exit(0)



    # hyperparameter `alpha`: threshold
    alpha = 0.01
    n = data_torch.shape[0]
    yf = torch.fft.rfft(data_torch)
    xf = torch.fft.rfftfreq(n, 0.001)
    yf_abs = np.abs(np.array(yf))
    print(yf.shape)
    print(yf_abs)
    max_val = yf_abs.max(axis=0)
    print(max_val)
    indices = yf_abs > 10
    print(indices.shape)

    indices = indices.astype(int)
    indices = torch.tensor(indices)
    print(indices)
    yf_clean = indices * yf
    new_yf_clean = torch.fft.irfft(yf_clean)


    plt.semilogy(xf, yf)
    plt.semilogy(xf, yf_clean)
    plt.xlabel("Hz")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    plt.plot(data_torch)
    plt.plot(new_yf_clean)
    plt.title("calcium data")
    plt.legend(["True", "FFT"])
    # fig2.savefig('derivative.png')
    plt.show()
    exit(0)




    # Add noise
    data = data[0:1000]
    n = len(data)
    data_noisy = data

    # # Plot true and noisy signal
    # fig1 = plt.figure()
    # plt.plot(data)
    # plt.plot(data_noisy)
    # plt.title("Signal")
    # plt.legend(["True", "Noisy"])
    # plt.show()
    # # exit(0)
    # Derivative with TVR
    diff_tvr = DiffTVR(n, 1)
    (deriv_tvr, _) = diff_tvr.get_deriv_tvr(
        data=data_noisy, deriv_guess=np.full(n + 1, 0.0), alpha=0.005, no_opt_steps=100
    )

    deriv_tvr = deriv_tvr[:-1]

    # Derivative with FFT
    # deriv_fft = fft(deriv_true, n)

    deriv_sf = savgol_filter(deriv_true, 5, 3, mode="nearest")

    # np.convolve
    def moving_average(interval, windowsize):
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, "same")
        return re

    deriv_con = moving_average(deriv_true, 5)

    # Plot TVR derivative
    fig2 = plt.figure()
    plt.plot(deriv_true)
    # plt.plot(deriv_tvr)
    # plt.plot(deriv_sf)
    plt.plot(deriv_con)
    plt.title("Derivative")
    plt.legend(["True", "CON"])
    # fig2.savefig('derivative.png')
    plt.show()
