import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_worm_data(single_worm_dataset, worm_name):
    """
    Plot the neural activity traces for some neurons in a given worm.
    single_worm_dataset: dict, the data for this worm.
    worm_name: str, name to give the worm.
    """
    # get the calcium data and neuron labels
    neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
    calcium_data = single_worm_dataset["named_data"]
    if len(neuron_to_idx) == 0:
        neuron_to_idx = single_worm_dataset["all_neuron_to_idx"]
        calcium_data = single_worm_dataset["all_data"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # filter for named neurons
    named_neurons = [key for key, val in idx_to_neuron.items() if not val.isnumeric()]
    if not named_neurons:
        named_neurons = list(idx_to_neuron.keys())
    # randomly select 10 neurons to plot
    inds = np.random.choice(named_neurons, 10)
    labels = [idx_to_neuron[i] for i in named_neurons]
    # plot calcium activity traces
    color = plt.cm.tab20(range(10))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color)
    plt.figure()
    plt.plot(calcium_data[:200, inds, 0])  # Ca traces in 1st dim
    plt.legend(
        labels,
        title="neuron ID",
        loc="upper right",
        fontsize="x-small",
        title_fontsize="small",
    )
    plt.xlabel("time")
    plt.ylabel("$\delta F / F$")
    plt.title(
        "{}: Calcium traces (first 200 timesteps) of 10 neurons".format(worm_name)
    )
    plt.show()
