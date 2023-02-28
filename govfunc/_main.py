#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _main.py
@time: 2023/2/28 12:15
'''

from data._main import *


def ctrl():
    config = OmegaConf.load("./conf/dataset.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)



    return 0


if __name__ == "__main__":

    ctrl()