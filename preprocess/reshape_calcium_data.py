import torch
import numpy as np
from utils import NEURONS_302


def reshape_calcium_data(single_worm_dataset):
    """
    Port calcium dataset to a new dataset
    with a slot for each of the 302 neurons.
    """
    # get the calcium data for this worm
    calcium_data = single_worm_dataset["all_data"]
    # get the neuron to idx map
    neuron_to_idx = single_worm_dataset["all_neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # get max time and number of neurons
    max_time = single_worm_dataset["max_time"]
    num_neurons = single_worm_dataset["num_neurons"]
    # load names of all 302 neurons
    neurons_302 = NEURONS_302
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
    # get the named neurons to indices mapping
    named_neuron_to_idx = dict(
        zip(
            np.array(neurons_302)[incl_neurons_mask.numpy()],
            np.where(incl_neurons_mask)[0],
        )
    )
    # display and return the reshaped data and a mask
    print(
        "\told data shape:",
        calcium_data.shape,
        "\n\tnew data shape:",
        new_calcium_data.shape,
    )
    return new_calcium_data, incl_neurons_mask, named_neuron_to_idx
