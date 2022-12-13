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
    calcium_data = single_worm_dataset['data']
    num_neurons = single_worm_dataset['num_neurons']
    neuron_ids = single_worm_dataset['neuron_ids']
    # filter for named neurons
    named_neurons = [key for key,val in neuron_ids.items() if not val.isnumeric()]
    if not named_neurons: named_neurons = list(neuron_ids.keys())
    # randomly select 10 neurons to plot
    inds = np.random.choice(named_neurons, 10)-1
    labels = [neuron_ids[i] for i in named_neurons]
    # plot calcium activity traces
    color = plt.cm.tab20(range(10))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color)
    plt.figure()
    plt.plot(calcium_data[:200, inds, 0]) # Ca traces in 1st dim
    plt.legend(labels, title='neuron ID', loc='upper right', 
            fontsize='x-small', title_fontsize='small')
    plt.xlabel('time')
    plt.ylabel('$\delta F / F$')
    plt.title('{}: Calcium traces (first 200 timesteps) of 10 neurons'.format(worm_name))
    plt.show()