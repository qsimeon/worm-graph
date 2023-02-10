"""
This script requires the `matplotlib` and a CSV file with evenly sampled data.
"""
FILENAME = "henon_a14b03.csv"
COLUMN = 0  ## Which column of csv file to use, 0 means leftmost.
POINTS = (
    -1
)  ## Number of points to use, more can be slower to render. -1 if all(but last).

E_DIMENSION = 3  ## Number of dimensions in embedding space -- 2 or 3.
TAU = 1  ## Delay, integer


import os
import csv
import matplotlib.pyplot as plt  ## pip install matplotlib
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    ## Read Data
    with open(os.path.join("analysis/takens", FILENAME), "r") as file:
        time_series = [float(row[COLUMN]) for row in csv.reader(file)][:POINTS]

    ## Process Data
    if E_DIMENSION == 2:
        delay_coordinates = [
            time_series[: -TAU if TAU else len(time_series)],  # t-T
            time_series[TAU:],  # t
        ]
    elif E_DIMENSION == 3:
        delay_coordinates = [
            time_series[TAU : -TAU if TAU else len(time_series)],  # t-T
            time_series[2 * TAU :],  # t
            time_series[: -2 * TAU if TAU else len(time_series)],  # t-2T
        ]
    else:
        raise ValueError(f"Invalid Embedding Dimension, '{E_DIMENSION}' not in [2, 3]!")

    ## Visualize Embedding
    fig = plt.figure()

    # Raw time series data
    ax = fig.add_subplot(211)
    ax.scatter(range(len(time_series[:100])), time_series[:100])
    ax.set_title("Discrete Time Series")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")

    # Reconstruction space
    ax = fig.add_subplot(212, projection="3d" if E_DIMENSION == 3 else None)
    ax.scatter(*delay_coordinates)
    ax.set_title(f"Embedding, Tau={TAU}")
    ax.set_xlabel(f"x(t-{TAU})")
    ax.set_ylabel("x(t)")
    if E_DIMENSION == 3:
        ax.set_zlabel(f"x(t-{2*TAU})")

    plt.show()
