"""
Tests the model optimization function `optimize_model`.
"""
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import NetworkLSTM
from data._main import get_dataset
from train._utils import optimize_model, model_predict

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # number of neurons we want to predict
    neuron_inds = [12, 13, 22, 25, 26]
    num_neurons = len(neuron_inds)
    # num_neurons = 1
    # number of split of data
    k = 2
    # offset of the prediction
    tau = 100
    # load a dataset (multiple worms)
    dataset = get_dataset(config)
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    # calcium_data = single_worm_dataset["calcium_data"][:, :num_neurons]
    calcium_data = single_worm_dataset["calcium_data"][:, neuron_inds]
    # named_neurons_mask = single_worm_dataset["named_neurons_mask"][:num_neurons]
    named_neurons_mask = single_worm_dataset["named_neurons_mask"][neuron_inds]
    # create a model
    model = NetworkLSTM(num_neurons, 64).double()
    # keyword args to `split_train_test`
    kwargs = dict(
        k_splits=k,
        seq_len=10,
        batch_size=128,
        train_size=1646,
        test_size=1646,
        # TODO: Why does `shuffle=True` improve performance so much?
        shuffle=True,
        reverse=False,
        tau=tau,
    )
    # train the model with the `optimize_model` function
    model, log = optimize_model(
        calcium_data,
        model,
        mask=named_neurons_mask,
        num_epochs=10,
        learn_rate=0.1,
        **kwargs,
    )
    # make predictions with trained model
    targets, predictions = model_predict(model, calcium_data * named_neurons_mask)
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")
    # plot entered loss curves
    plt.figure()
    plt.plot(log["epochs"], log["centered_train_losses"], label="train")
    plt.plot(log["epochs"], log["centered_test_losses"], label="test")
    plt.legend()
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss - Baseline")
    plt.show()
    # figures of neuron calcium target and prediction
    for neuron in range(num_neurons):
        plt.figure()
        plt.plot(targets[:, neuron], label="target")
        plt.plot(predictions[:, neuron], alpha=0.8, label="prediction")
        plt.legend(loc="upper right")
        plt.title("Neuron %s target and prediction" % neuron)
        plt.xlabel("Time")
        plt.ylabel("$Ca^{2+} \Delta F / F$")
        plt.show()
