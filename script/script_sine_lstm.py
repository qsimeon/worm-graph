#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: script_sine_lstm.py
@time: 2023/3/14 11:21
'''

from train._utils import *


model = get_model(OmegaConf.load("conf/model.yaml"))
dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))

print(dataset["worm0"].keys())

plt.plot(dataset["worm0"]["calcium_data"][:, 0])

plt.plot(dataset["worm0"]["residual_calcium"][:, 0])
plt.show()

