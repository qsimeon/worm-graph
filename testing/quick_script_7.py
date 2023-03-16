"""
Tests the `split_train_test` function interleaved 
cuts of train and test data given calcium data
from a single worm.
"""

import numpy as np
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
    # create train and test data loaders with `split_train_test`
    kwargs = dict(
        k_splits=3,
        seq_len=999,
        batch_size=63,
        train_size=1024,
        test_size=512,
        shuffle=False,
        reverse=False,
        tau=100,
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
    b = 0 # batch
    n = np.random.choice(302) # neuron
    # multiple inputs
    plt.figure()
    plt.plot(train_mask.to(float).numpy(), color='black', label="train mask") # train mask
    plt.plot(test_mask.to(float).numpy(), color='darkgray', label="test mask") # test mask
    # train
    plt.plot(metadata_train['time_vec'][b,:], X_train[b, :, n], color='red', label='1st train input') # first sample in train batch
    plt.plot(metadata_train['time_vec'][b,:], 0.15*np.random.rand() + X_train[b+1, :, n], color='orange', label='2nd train input') # second sample in train batch
    plt.plot(metadata_train['time_vec'][b,:], 0.2*np.random.rand() + X_train[b+2, :, n], color='chocolate', label='3rd train input') # second sample in train batch
    # test
    plt.plot(metadata_test['time_vec'][b, :], X_test[b, :, n], color='blue', label='1st test input') # first sample in test batch, neuron 0
    plt.plot(metadata_test['time_vec'][b, :], 0.15*np.random.rand() + X_test[b+1, :, n], color='cyan', label='2nd test input') # second sample in test batch
    plt.plot(metadata_test['time_vec'][b, :], 0.2*np.random.rand() + X_test[b+2, :, n], color='limegreen', label='3rd test input') # second sample in test batch
    # labeling
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend(loc='lower right')
    plt.title("samples from batch %s, neuron %s"%(b,n))
    plt.show()
    # inputs and targets
    plt.figure()
    plt.plot(train_mask.to(float).numpy(), color='black', label="train mask") # train mask
    plt.plot(test_mask.to(float).numpy(), color='darkgray', label="test mask") # test mask
    # train
    plt.plot(metadata_train['time_vec'][b, :], X_train[b, :, n], color='red', label='1st train input') # first sample in train batch
    plt.plot(metadata_train['tau'][b] + metadata_train['time_vec'][b, :], 0.2*np.random.rand() + Y_train[b, :, n], color='orange', label='1st train target') # second sample in train batch
    # test
    plt.plot(metadata_test['time_vec'][b, :], X_test[b, :, n], color='blue', label='1st test input') # first sample in test batch
    plt.plot(metadata_test['tau'][b] + metadata_test['time_vec'][b, :], 0.2*np.random.rand() + Y_test[b, :, n], color='cyan', label='1st test target') # second sample in test batch
    # labeling
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend(loc='lower right')
    plt.title("samples from batch %s, neuron %s"%(b,n))
    plt.show()
