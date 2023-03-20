"""
Tests the model optimization function `optimize_model`.
"""

import matplotlib.pyplot as plt
from models._utils import NetworkLSTM
from data._utils import load_sine
from train._utils import split_train_test, optimize_model, model_predict


if __name__ == "__main__":
    # pick indices of neurons we want
    neuron_inds = range(0, 1)
    num_neurons = len(neuron_inds)
    # load a dataset (multiple worms)
    dataset = load_sine()
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"][:, neuron_inds]
    named_neurons_mask = single_worm_dataset["named_neurons_mask"][neuron_inds]
    time_in_seconds = single_worm_dataset.get("time_in_seconds", None)
    # create a model
    model = NetworkLSTM(num_neurons, 64).double()
    # keyword args to `split_train_test`
    tau_in = 1
    kwargs = dict(
        k_splits=2,
        seq_len=10,
        batch_size=128,
        train_size=1654,
        test_size=1654,
        time_vec=time_in_seconds,
        # TODO: Why does `shuffle=True` improve performance so much?
        shuffle=True,
        reverse=False,
        tau=tau_in,
    )
    # create data loaders and train/test masks
    train_loader, test_loader, train_mask, test_mask = split_train_test(
        calcium_data,
        **kwargs,
    )
    # train the model with the `optimize_model` function
    model, log = optimize_model(
        model,
        train_loader,
        test_loader,
        neurons_mask=named_neurons_mask,
        num_epochs=50,
        learn_rate=0.01,
    )
    # make predictions with trained model
    tau_out = 50
    targets, predictions = model_predict(
        model,
        calcium_data * named_neurons_mask,
        tau=tau_out,
    )
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")
    # plot entered loss curves
    plt.figure()
    plt.plot(log["epochs"], log["centered_train_losses"], label="train")
    plt.plot(log["epochs"], log["centered_test_losses"], label="test")
    plt.legend(loc="best")
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss - Baseline")
    plt.show()
    # figures of neuron calcium target and prediction
    for neuron in range(num_neurons):
        plt.figure()
        plt.plot(time_in_seconds, targets[:, neuron], label="target")
        plt.plot(time_in_seconds, predictions[:, neuron], alpha=0.8, label="prediction")
        plt.legend()
        plt.title("Neuron %s target and prediction ($\\tau = %s$)" % (neuron, tau_out))
        plt.xlabel("Time (seconds)")
        plt.ylabel("$Ca^{2+} \Delta F / F$")
        plt.show()
