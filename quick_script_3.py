"""
Check whether data loaders work as intended. 
"""

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from data._main import get_dataset
from train._utils import split_train_test


if __name__ == "__main__":
    # load dataset and get calcium data for one worm
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    calcium_data = dataset["worm0"]["calcium_data"]
    # create train and test data loaders
    kwargs = dict(tau=1, shuffle=False, reverse=False)
    train_loader, test_loader, train_mask, test_mask = split_train_test(
        calcium_data,
        seq_len=[7, 13, 101],
        k_splits=2,
        train_size=131702,
        test_size=131702,
        **kwargs,
    )

    print("Train batches")
    for data in train_loader:
        X, Y, meta = data
        print(X.shape)
    print()

    print("Test batches")
    for data in test_loader:
        X, y, meta = data
        print(X.shape)
    print()

    plt.figure()

    plt.plot(train_mask.to(float).numpy(), label="train")

    plt.plot(test_mask.to(float).numpy(), label="test")

    plt.legend()

    plt.title("Train and Test Masks")

    plt.xlabel("Time")

    plt.ylabel("Test (0) / Train (1)")

    plt.show()
