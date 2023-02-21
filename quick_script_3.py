from data._main import get_dataset

from omegaconf import OmegaConf

from train._utils import split_train_test

import matplotlib.pyplot as plt

dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))

calcium_data = dataset["worm0"]["calcium_data"]

kwargs = dict(tau=1, shuffle=True, reverse=True)

train_loader, test_loader, train_mask, test_mask = split_train_test(
    calcium_data,
    seq_len=[7, 13, 101],
    k_splits=5,
    train_size=3312 * 3,
    test_size=3312 * 3,
    **kwargs,
)

print("Train batches")
for data in train_loader:
    X, Y, meta = data
    print(X.shape)
    # break
print()

print("Test batches")
for data in test_loader:
    X, y, meta = data
    print(X.shape)
    # break
print()

plt.figure()

plt.plot(train_mask.to(float).numpy(), label="train")

plt.plot(test_mask.to(float).numpy(), label="test")

plt.legend()

plt.title("Train and Test Masks")

plt.xlabel("Time")

plt.ylabel("Test (0) / Train (1)")

plt.show()
