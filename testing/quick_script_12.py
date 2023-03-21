"""
make seq_len == tau
use predictions to predict next seq_len
"""

import matplotlib.pyplot as plt
from models._utils import NetworkLSTM
from data._utils import *
from train._utils import split_train_test, optimize_model, model_predict
from visualization._main import *

if __name__ == "__main__":
    # pick indices of neurons we want
    neuron_inds = range(0, 302)
    num_neurons = len(neuron_inds)
    # decide tau as the offset of the target
    tau_train = 100
    tau_test = 100
    # load a dataset (multiple worms)
    dataset = load_Uzel2022()
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]

    calcium_data = single_worm_dataset["calcium_data"][:, neuron_inds]
    named_neurons_mask = single_worm_dataset["named_neurons_mask"][neuron_inds]
    time_vec = single_worm_dataset.get("time_in_seconds", None)
    # create a model
    model = NeuralCFC(num_neurons, 100, 2).double()
    # keyword args to `split_train_test`
    kwargs = dict(
        k_splits=2,
        seq_len=tau_train,
        batch_size=128,
        train_size=1654,
        test_size=1654,
        time_vec=time_vec,
        # TODO: Why does `shuffle=True` improve performance so much?
        shuffle=True,
        reverse=False,
        tau=tau_train,
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
        num_epochs=100,
        learn_rate=0.01,
    )
    # make predictions with trained model
    # use predictions to predict next predictions

    targets, predictions = model_predict(model, calcium_data * named_neurons_mask)
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")

    for neuron in range(0, 60):
        if single_worm_dataset["named_neurons_mask"][neuron].item():
            plt.figure()
            plt.plot(range(targets.shape[0]), targets[:, neuron], label="target")
            plt.plot(range(tau_test, tau_test + predictions.shape[0]), predictions[:, neuron], alpha=0.8, label="prediction")
            plt.legend()
            plt.title("Neuron %s target and prediction" % neuron)
            plt.xlabel("Time")
            plt.ylabel("$Ca^{2+} \Delta F / F$")
            plt.show()
