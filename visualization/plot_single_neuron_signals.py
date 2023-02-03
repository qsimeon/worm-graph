import torch
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots


def plot_single_neuron_signals(single_worm_dataset, neuron_idx):
    """
    Visualizes the full Ca2+ recording, the residual of the former,
    and the 20-lag partial autocorrelation function of the specified
    neuron in the worm.
    """
    neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
    calcium_data = single_worm_dataset["named_data"]
    if len(neuron_to_idx) == 0:
        neuron_to_idx = single_worm_dataset["all_neuron_to_idx"]
        calcium_data = single_worm_dataset["all_data"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    max_time = single_worm_dataset["max_time"]
    neuron = idx_to_neuron[neuron_idx]
    # plot the full Ca2+ recording of that neuron
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].plot(calcium_data[:, neuron_idx])
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("normalized $\Delta F/F$")
    axs[0].set_title(
        "Signal: Calcium activity, \nNeuron %s, Recording time: %s" % (neuron, max_time)
    )
    # plot the residuals for the full time series
    residuals = torch.diff(
        calcium_data[:, neuron_idx].squeeze(), prepend=calcium_data[0, neuron_idx]
    )
    mean = residuals.mean().item()
    print("mean residual:", mean)
    print()
    axs[1].plot(residuals)
    axs[1].axhline(mean, color="k", linewidth=3, label="mean")
    axs[1].set_xlabel("Time $t$")
    axs[1].set_ylabel("residual")
    axs[1].set_title("Residuals of Ca2+ signal of neuron %s" % neuron)
    axs[1].legend()
    # plot the autocorrelation function of that neuron's Ca2+ signal
    tsaplots.plot_pacf(calcium_data[:, neuron_idx].squeeze(), ax=axs[2], lags=20)
    axs[2].set_title("Partial autocorrelation of neuron %s" % neuron)
    axs[2].set_xlabel("Lag at tau τ")
    axs[2].set_ylabel("correlation coefficient")
    plt.show()
