#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: smooth_method_trial.py
@time: 2023/3/3 16:36
"""

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


if __name__ == "__main__":

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


    n = data.shape[0]

    print((data[1:]-data[:n-1]).shape)



