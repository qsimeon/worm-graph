#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@file: data_process_celegans_atlas.py
@software: PyCharm 2022.3
@time: 2023/2/23 19:48
@desc: reformat data from "exported_data.tar.gz" into dict{ 'neuro_name': 'Calcium records on time series'}
"""

import tarfile
import os
import numpy as np
import pickle


# A^{T}
def transpose_2d(data):
    transposed = []
    for i in range(len(data[0])):
        new_row = []
        for row in data:
            new_row.append(row[i])
        transposed.append(new_row)
    return transposed


# extract .tar.gz file
filename = "exported_data_unc31"
tf = tarfile.open(filename + ".tar.gz")
tf.extractall(filename)





gcamp = []
label = []
gcamp_np = []

num = 49
for i in range(0, num):
    with open(filename + "/" + str(i) + "_gcamp.txt", "r") as f:
        cal_list = []
        for line in f.readlines():
            cal = list(map(float, line.split(" ")))
            cal_list.append(cal)
        cal_tmp = transpose_2d(cal_list)

    with open("./" + filename + "/" + str(i) + "_labels.txt", "r") as f:
        label_list = []
        for line in f.readlines():
            l = line.strip("\n")
            label_list.append(l)

    cal_noNull = []
    l_noNull = []
    # subtract data without label
    for j in range(0, len(label_list)):
        if label_list[j] != "":
            cal_noNull.append(cal_tmp[j])
            l_noNull.append(label_list[j])

    cal_noNull_array = np.array(cal_noNull)
    gcamp_np.append(dict(zip(l_noNull, cal_noNull_array)))
print("------data reformatted------")
print(str(num) + " txt have been reformatted.")

file = open("gcamp.pickle", "wb")
pickle.dump(gcamp_np, file)
file.close()
print("------save to Pickle------")
