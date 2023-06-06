"""
Tests whether the dataset and 
data loader generates samples 
and batches, respectively, as expected.
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data._utils import NeuralActivityDataset, load_sine_noise

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = load_sine_noise()
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]
    # create a Pytorch dataset from `calcium_data`
    neural_dataset = NeuralActivityDataset(
        calcium_data,
        seq_len=999,
        num_samples=128,
        tau=400,  # target offset
        reverse=False,
    )
    # create dataloader from neural dataset
    loader = DataLoader(
        neural_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )
    # display shapes of batches
    print("Batches")
    for data in loader:
        X, Y, metadata = data
        print(X.shape)
        print()
    # Last sample in last batch, input and target from neuron 0
    plt.figure()
    plt.plot(
        metadata["time_vec"][-1, :],
        X[-1, :, 0],
        label="input",
    )
    plt.plot(
        metadata["time_vec"][-1, :] + metadata["tau"][-1],
        0.1 * np.random.randn() + Y[-1, :, 0],
        label="target",
    )
    plt.xlabel("Time")
    plt.ylabel("$Ca^{2+}$ ($\Delta F / F$)")
    plt.title("Last sample, Last batch, Neuron 0, Input & Target")
    plt.legend()
    plt.show()
