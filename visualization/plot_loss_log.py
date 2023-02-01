import matplotlib.pyplot as plt
import numpy as np


def plot_loss_log(log, plt_title=""):
    """
    Plot the loss curves returned from `optimize_model`.
    """
    plt.figure()
    plt.plot(
        log["epochs"],
        np.log10(log["train_losses"]),
        label="train",
        color="r",
        linewidth=2,
    )
    plt.plot(
        log["epochs"],
        np.log10(log["test_losses"]),
        label="test",
        color="b",
        linewidth=2,
    )
    plt.plot(
        log["epochs"],
        np.log10(log["base_train_losses"]),
        label="train baseline",
        color="r",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        log["epochs"],
        np.log10(log["base_test_losses"]),
        label="test baseline",
        color="b",
        linestyle="--",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("log MSE")
    plt.legend(labelspacing=0)
    plt.title(plt_title)
    plt.show()
