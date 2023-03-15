"""
Tests whether the data loaders generate  
batches and samples as expected.
"""

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data._main import get_dataset
from data._utils import NeuralActivityDataset
from train._utils import split_train_test
from numpy import random


if __name__ == "__main__":
    # load dataset and get calcium data for one worm
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    calcium_data = dataset["worm0"]["calcium_data"]
    # create a Pytorch dataset from `calcium_data`
    neural_dataset = NeuralActivityDataset(
        calcium_data,
        seq_len=311,
        num_samples=1024,
        reverse=True,
    )
    # create dataloader from neural dataset
    loader = DataLoader(
        neural_dataset,
        batch_size=127,
        shuffle=False,
        pin_memory=True,
    )
    # display shapes of batches
    print("Batches")
    for data in loader:
        X, Y, metadata = data
        print(X.shape)
        print()
    # figure of first samples from neuron 0 in last batch
    plt.figure()
    for i in range(5):
        plt.plot(
            metadata["time_vec"][i, :],
            0.5 * random.randn() + X[i, :, 0],
            label="sample %s" % i,
        )
    plt.xlabel("Time")
    plt.ylabel("$Ca^{2+} \Delta F / F$")
    plt.title("Last batch, Neuron 0, First 5 samples")
    plt.legend()
    plt.show()
    # create train and test data loaders using
    kwargs = dict(
        k_splits=2,
        seq_len=101,
        batch_size=63,
        train_size=512,
        test_size=256,
        shuffle=True,
        reverse=False,
    )
    train_loader, test_loader, train_mask, test_mask = split_train_test(
        calcium_data,
        **kwargs,
    )
    # shapes of train batches
    print("Train batches")
    for data in train_loader:
        X, Y, metadata = data
        print(X.shape)
    print()
    # shapes of test batches
    print("Test batches")
    for data in test_loader:
        X, y, metadata = data
        print(X.shape)
    print()
    # display figure of train/test masks
    plt.figure()
    plt.plot(train_mask.to(float).numpy(), label="train")
    plt.plot(test_mask.to(float).numpy(), label="test")
    plt.legend()
    plt.title("Train and Test Masks")
    plt.xlabel("Time")
    plt.ylabel("Test (0) / Train (1)")
    plt.show()
