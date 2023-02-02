import os
import torch
import pandas as pd
import numpy as np
from utils import ROOT_DIR


def reshape_calcium_data(single_worm_dataset):
    """
    Port calcium dataset to a new dataset
    with a slot for each of the 302 neurons.
    """
    # get the calcium data for this worm
    calcium_data = single_worm_dataset["data"]
    # get the neuron to idx map
    neuron_to_idx = single_worm_dataset["neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # get max time and number of neurons
    max_time = single_worm_dataset["max_time"]
    num_neurons = single_worm_dataset["num_neurons"]
    # load names of all 302 neurons.
    neurons_302 = sorted(
        pd.read_csv(
            os.path.join(ROOT_DIR, "data", "raw", "neurons_302.txt"),
            sep=" ",
            header=None,
            names=["neuron"],
        ).neuron
    )
    # check the calcium data
    assert len(idx_to_neuron) == calcium_data.size(
        1
    ), "Number of neurons in calcium dataset does not match number of recorded neurons"
    # assign each neuron to a particular slot
    slot_to_neuron = dict((k, v) for k, v in enumerate(neurons_302))
    neuron_to_slot = dict((v, k) for k, v in enumerate(neurons_302))
    # create the new calcium data structure
    new_calcium_data = torch.zeros(max_time, 302, 1, dtype=calcium_data.dtype)
    # create a mask of which neurons have data
    incl_neurons_mask = torch.zeros(302, dtype=torch.bool)
    # fill the new data structure and mask
    for slot, neuron in slot_to_neuron.items():
        if neuron in neuron_to_idx:
            idx = neuron_to_idx[neuron]
            new_calcium_data[:, slot, :] = calcium_data[:, idx, :]
            incl_neurons_mask[slot] = True
    # display and return the reshaped data and a mask
    print(
        "\told data shape:",
        calcium_data.shape,
        "\n\tnew data shape:",
        new_calcium_data.shape,
    )
    return new_calcium_data, incl_neurons_mask
