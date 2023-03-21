
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.cluster import hierarchy

import sklearn

from train._main import *


def plot_coefficient(data):
    plt.figure(figsize=(20, 20))
    sns.heatmap(data=data, square=True, cmap="RdBu_r", center=0, linecolor='grey', linewidths=0.3)
    plt.show()



worm = "worm0"
data = pd.read_hdf("./govfunc/coefficient/coef_" + worm + ".hdf")

print(data)
plot_coefficient(data)


