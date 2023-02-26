#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: plot_neuroResponds.py
@time: 2023/2/24 15:27
@desc: plot neuro responds in Figure 1d
'''

import tarfile
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = open('gcamp.pickle', 'rb')
gcamp_np = pickle.load(file)

########### hyperparameters ##############
# which stimulation is wanted to plot
wanted = 2



g = gcamp_np[wanted]
# to dataframe for further process
gcal = pd.DataFrame.from_dict(g, orient='index')
gcal.sort_index(inplace=True)

##### NOTE: we just need to show how different neurons respond to the stimulation
##### thus the numerical value can be normalized.

# data normalization: z-scoring
cnt = 0
interval = 5
for i in range(0, len(gcal.index)):
    gcal.iloc[i] = (gcal.iloc[i] - gcal.iloc[i].mean()) / gcal.iloc[i].std()
    gcal.iloc[i] = gcal.iloc[i] + cnt
    cnt -= interval

# start plotting
plt.figure(figsize=(6, 6))
axe = plt.gca()
axe.spines['top'].set_color('none')
axe.spines['right'].set_color('none')
axe.spines['left'].set_color('none')
plt.axis([0, gcal.shape[1], -len(gcal.index) * interval, 1])

# transfer to list in order to fit in ticks-reformat
list_y = []
list_label = []
for i in range(0, len(gcal.index) * interval, interval):
    list_y.append(-i)

for j in range(0, len(gcal.index)):
    list_label.append(gcal.index[j])

plt.ylabel("Responding Neuron")
plt.xlabel("Time(s)")

plt.yticks(list_y, list_label, fontproperties='Times New Roman', size=4)

for i in range(0, len(gcal.index)):
    plt.plot(range(0, gcal.shape[1]), gcal.iloc[i], color=sns.color_palette("deep", n_colors=20)[i % 20], linewidth=0.3)


plt.savefig('./neuro_response_No' + str(wanted) + '.png', dpi=800, bbox_inches='tight')
plt.show()
