"""
Tests the model optimization function `optimize_model`.
"""
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import LinearNN, NeuralCFC, NetworkLSTM
from data._main import get_dataset
from train._utils import optimize_model, model_predict

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # number of neurons we want to predict
    num_neurons = 1
    # number of split of data
    k = 2
    # load a dataset (multiple worms)
    dataset = get_dataset(config)
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"][:, :num_neurons]
    named_neurons_mask = single_worm_dataset["named_neurons_mask"][:num_neurons]
    # create a model
    # model = LinearNN(num_neurons, 64).double()
    # model = NeuralCFC(num_neurons, 64).double()
    model = NetworkLSTM(num_neurons, 64).double()
    # keyword args to `split_train_test`
    kwargs = dict(
        k_splits=k,
        seq_len=10,
        batch_size=128,
        train_size=1024,
        test_size=1024,
        # TODO: Why does `shuffle=True` improve performance so much?
        shuffle=True,
        reverse=False,
        tau=1,
    )
    # train the model with the `optimize_model` function
    model, log = optimize_model(
        calcium_data,
        model,
        mask=named_neurons_mask,
        num_epochs=10,
        learn_rate=0.01,
        **kwargs,
    )
    # keyword args to `model_predict`
    # kwargs = dict(tau=1)
    kwargs = dict(tau=len(calcium_data) // k)
    # make predictions with trained model
    targets, predictions = model_predict(
        model, calcium_data * named_neurons_mask, **kwargs
    )
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
    # figure of neuron 0 calcium target and prediction
    plt.figure()
    plt.plot(targets[:, 0], label="target")
    plt.plot(predictions[:, 0], alpha=0.8, label="prediction")
    plt.legend()
    plt.title("Neuron 0 target and prediction")
    plt.xlabel("Time")
    plt.ylabel("$Ca^{2+} \Delta F / F$")
    plt.show()
    # # figure of neuron 49 calcium target and prediction
    # plt.figure()
    # plt.plot(targets[:, 49], label="target")
    # plt.plot(predictions[:, 49], alpha=0.8, label="prediction")
    # plt.legend()
    # plt.title("Neuron 49 target and prediction")
    # plt.xlabel("Time")
    # plt.ylabel("$Ca^{2+} \Delta F / F$")
    # plt.show()
    # # figure of neuron 60 calcium target and prediction
    # plt.figure()
    # plt.plot(targets[:, 60], label="target")
    # plt.plot(predictions[:, 60], alpha=0.8, label="prediction")
    # plt.legend()
    # plt.title("Neuron 60 target and prediction")
    # plt.xlabel("Time")
    # plt.ylabel("$Ca^{2+} \Delta F / F$")
    # plt.show()
    # # figure of neuron 200 calcium target and prediction
    # plt.figure()
    # plt.plot(targets[:, 200], label="target")
    # plt.plot(predictions[:, 200], alpha=0.8, label="prediction")
    # plt.legend()
    # plt.title("Neuron 200 target and prediction")
    # plt.xlabel("Time")
    # plt.ylabel("$Ca^{2+} \Delta F / F$")
    # plt.show()
    # # figure of neuron 300 calcium target and prediction
    # plt.figure()
    # plt.plot(targets[:, 300], label="target")
    # plt.plot(predictions[:, 300], alpha=0.8, label="prediction")
    # plt.legend()
    # plt.title("Neuron 300 target and prediction")
    # plt.xlabel("Time")
    # plt.ylabel("$Ca^{2+} \Delta F / F$")
    # plt.show()
