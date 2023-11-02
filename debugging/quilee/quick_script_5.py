"""
Tests the splitting of the calcium data from
a single worm into interleaved chunks of train 
and test using the `split_train_test` function.
"""

import numpy as np
import matplotlib.pyplot as plt
from data._utils import load_sine_seq
from train._utils import split_train_test

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = load_sine_seq()
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]
    time_vec = single_worm_dataset["time_in_seconds"]
    # create train and test data loaders with `split_train_test`
    kwargs = dict(
        k_splits=2,
        seq_len=1000,
        batch_size=16,
        train_size=128,
        test_size=128,
        time_vec=time_vec,
        shuffle=True,  # shuffled train samples
        reverse=False,
        tau=500,
        use_residual=False,
    )
    train_loader, test_loader, train_mask, test_mask = split_train_test(
        calcium_data,
        **kwargs,
    )
    # shapes of train batches
    print("Train batches")
    for X_train, Y_train, metadata_train in train_loader:
        break
    print(X_train.shape, Y_train.shape)
    # shapes of test batches
    print("Test batches")
    for X_test, Y_test, metadata_test in test_loader:
        break
    print(X_test.shape, Y_test.shape)
    # display figures on inputs, targets and masks
    b = 0  # batch index
    n = 0  # neuron index
    # multiple inputs
    plt.figure()
    # train
    plt.plot(
        metadata_train["time_vec"][b, :],
        X_train[b, :, n],
        color="red",
        label="1st train input",
    )  # first sample in train batch
    plt.plot(
        metadata_train["time_vec"][b + 1, :],
        0.15 * np.random.rand() + X_train[b + 1, :, n],
        color="orange",
        label="2nd train input",
    )  # second sample in train batch
    plt.plot(
        metadata_train["time_vec"][b + 2, :],
        0.2 * np.random.rand() + X_train[b + 2, :, n],
        color="chocolate",
        label="3rd train input",
    )  # second sample in train batch
    # test
    plt.plot(
        metadata_test["time_vec"][b, :],
        X_test[b, :, n],
        color="blue",
        label="1st test input",
    )  # first sample in test batch, neuron 0
    plt.plot(
        metadata_test["time_vec"][b + 1, :],
        0.15 * np.random.rand() + X_test[b + 1, :, n],
        color="cyan",
        label="2nd test input",
    )  # second sample in test batch
    plt.plot(
        metadata_test["time_vec"][b + 2, :],
        0.2 * np.random.rand() + X_test[b + 2, :, n],
        color="limegreen",
        label="3rd test input",
    )  # second sample in test batch
    # train mask
    ylo, yhi = plt.gca().get_ylim()
    plt.gca().fill_between(
        np.arange(len(calcium_data)),
        ylo,
        yhi,
        where=train_mask,
        alpha=0.1,
        facecolor="cyan",
        label="train mask",
    )
    # test mask
    plt.gca().fill_between(
        np.arange(len(calcium_data)),
        ylo,
        yhi,
        where=test_mask,
        alpha=0.1,
        facecolor="magenta",
        label="test mask",
    )
    # labeling
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.legend(loc="best")
    plt.title("samples from batch %s, neuron %s" % (b, n))
    plt.show()
    # inputs and targets
    plt.figure()
    # train
    plt.plot(
        metadata_train["time_vec"][b, :],
        X_train[b, :, n],
        color="red",
        label="1st train input",
    )  # first sample in train batch
    plt.plot(
        metadata_train["tau"][b] + metadata_train["time_vec"][b, :],
        0.2 * np.random.rand() + Y_train[b, :, n],
        color="orange",
        label="1st train target",
    )  # second sample in train batch
    # test
    plt.plot(
        metadata_test["time_vec"][b, :],
        X_test[b, :, n],
        color="blue",
        label="1st test input",
    )  # first sample in test batch
    plt.plot(
        metadata_test["tau"][b] + metadata_test["time_vec"][b, :],
        0.2 * np.random.rand() + Y_test[b, :, n],
        color="cyan",
        label="1st test target",
    )  # second sample in test batch
    # train mask
    ylo, yhi = plt.gca().get_ylim()
    plt.gca().fill_between(
        np.arange(len(calcium_data)),
        ylo,
        yhi,
        where=train_mask,
        alpha=0.1,
        facecolor="cyan",
        label="train mask",
    )
    # test mask
    plt.gca().fill_between(
        np.arange(len(calcium_data)),
        ylo,
        yhi,
        where=test_mask,
        alpha=0.1,
        facecolor="magenta",
        label="test mask",
    )
    # labeling
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.legend(loc="best")
    plt.title("samples from batch %s, neuron %s" % (b, n))
    plt.show()
