import torch
import matplotlib.pyplot as plt
from statsmodels import tsaplots


def plot_single_neuron_signals(single_worm_dataset, neuron_idx):
    """
    Visualizes the full Ca2+ recording, the residual of the former,
    and the 20-lag partial autocorrelation function of the specified
    neuron in the worm.
    """
    calcium_data = single_worm_dataset["data"]
    neuron_id = single_worm_dataset["neuron_id"]
    id_neuron = dict((v, k) for k, v in neuron_id.items())
    max_time = single_worm_dataset["max_time"]
    # selecting a neuron to plot
    idx = neuron_idx
    nid = id_neuron[idx] 
    print ("hello " ) 
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # plot the full Ca2+ recording of that neuron
    axs[0].plot(calcium_data[:, idx])
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("normalized $\Delta F/F$")
    axs[0].set_title(
        "Signal: Calcium activity, \nNeuron %s, Recording time: %s" % (nid, max_time)
    )
    # plot the residuals for the full time series
    residuals = torch.diff(calcium_data[:, idx].squeeze(), prepend=calcium_data[0, idx])
    mean = residuals.mean().item()
    print("mean residual:", mean)
    print()
    axs[1].plot(residuals)
    axs[1].axhline(mean, color="k", linewidth=3, label="mean")
    axs[1].set_xlabel("Time $t$")
    axs[1].set_ylabel("residual")
    axs[1].set_title("Residuals of Ca2+ signal of neuron %s" % nid)
    axs[1].legend()
    # plot the autocorrelation function of that neuron's Ca2+ signal
    tsaplots.plot_pacf(calcium_data[:, idx].squeeze(), ax=axs[2], lags=20)
    axs[2].set_title("Partial autocorrelation of neuron %s" % nid)
    axs[2].set_xlabel("Lag at tau τ")
    axs[2].set_ylabel("correlation coefficient")
    plt.show()
