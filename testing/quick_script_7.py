"""
Tests the `split_train_test` function interleaved 
cuts of train and test data given calcium data
from a single worm.
"""

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from data._main import get_dataset
from train._utils import split_train_test

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = get_dataset(config)
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]
    # create train and test data loaders using
    kwargs = dict(
        k_splits=3,
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
