import matplotlib.pyplot as plt


def plot_target_prediction(target, prediction, plt_title=""):
    """
    Make a plot of prediction versus target for the full trace
    of a single neuron.
    """
    max_time = len(target)
    plt.figure()
    plt.plot(target, linestyle="-", label="target", linewidth=2)
    plt.plot(prediction, linestyle=":", label="prediction", linewidth=3)
    plt.axvline(x=max_time // 2, c="r", linestyle="--", linewidth=4)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Residual $\Delta F/F$")
    plt.title(plt_title)
    plt.show()
