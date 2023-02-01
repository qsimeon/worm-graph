import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_hidden_experiment(hidden_experiment):
    """
    Plot the results from the logs returned in `hidden_experiment`
    by `lstm_hidden_size_experiment`.
    """
    color = plt.cm.YlGnBu(np.linspace(0, 1, len(hidden_experiment) + 2))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color[2:])
    fig, axs = plt.subplots(1, 2)

    for hs in sorted(hidden_experiment):
        axs[0].plot(
            hidden_experiment[hs]["epochs"],
            np.log10(hidden_experiment[hs]["train_losses"]),
            label="hidden_size=%d" % hs,
            linewidth=2,
        )

    axs[0].plot(
        hidden_experiment[hs]["epochs"],
        np.log10(hidden_experiment[hs]["base_train_losses"]),
        linewidth=2,
        color="r",
        label="baseline",
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("log MSE")
    axs[0].set_title("Training")

    plt.gca().set_prop_cycle(None)
    for hs in sorted(hidden_experiment):
        axs[1].plot(
            hidden_experiment[hs]["epochs"],
            np.log10(hidden_experiment[hs]["test_losses"]),
            label="hidden_size=%d" % hs,
            linewidth=2,
        )

    axs[1].plot(
        hidden_experiment[hs]["epochs"],
        np.log10(hidden_experiment[hs]["base_test_losses"]),
        color="r",
        label="baseline",
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_yticks(axs[0].get_yticks())
    axs[1].legend(loc="upper right", borderpad=0, labelspacing=0)
    axs[1].set_title("Validation")

    fig.suptitle("LSTM network model loss curves with various hidden units")
    plt.show()
