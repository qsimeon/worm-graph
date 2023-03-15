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

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # load dataset and get calcium data for one worm
    dataset = get_dataset(config)
    calcium_data = dataset["worm0"]["calcium_data"]
    # create a Pytorch dataset from `calcium_data`
    neural_dataset = NeuralActivityDataset(
        calcium_data,
        seq_len=517,
        num_samples=1024,
        tau=100,  # offset of target
        reverse=False,
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
    # Last sample in last batch, input and target from neuron 0
    plt.figure()
    plt.plot(
        metadata["time_vec"][-1, :],
        X[-1, :, 0],
        label="input",
    )
    plt.plot(
        metadata["time_vec"][-1, :] + metadata["tau"][-1],
        0.5 * random.randn() + Y[-1, :, 0],
        label="target",
    )
    plt.xlabel("Time")
    plt.ylabel("$Ca^{2+} \Delta F / F$")
    plt.title("Last sample, Last batch, Neuron 0 input & target")
    plt.legend()
    plt.show()
    # # create train and test data loaders using
    # kwargs = dict(
    #     k_splits=2,
    #     seq_len=101,
    #     batch_size=63,
    #     train_size=512,
    #     test_size=256,
    #     shuffle=True,
    #     reverse=False,
    # )
    # train_loader, test_loader, train_mask, test_mask = split_train_test(
    #     calcium_data,
    #     **kwargs,
    # )
    # # shapes of train batches
    # print("Train batches")
    # for data in train_loader:
    #     X, Y, metadata = data
    #     print(X.shape)
    # print()
    # # shapes of test batches
    # print("Test batches")
    # for data in test_loader:
    #     X, y, metadata = data
    #     print(X.shape)
    # print()
    # # display figure of train/test masks
    # plt.figure()
    # plt.plot(train_mask.to(float).numpy(), label="train")
    # plt.plot(test_mask.to(float).numpy(), label="test")
    # plt.legend()
    # plt.title("Train and Test Masks")
    # plt.xlabel("Time")
    # plt.ylabel("Test (0) / Train (1)")
    # plt.show()
