"""
Tests the model optimization function `optimize_model`.
"""
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import LinearNN
from data._main import get_dataset
from train._utils import optimize_model, model_predict

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = get_dataset(config)
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]
    # create a model
    model = LinearNN(302, 64).double()
    # keyword args to `split_train_test`
    kwargs = dict(
        k_splits=2,
        seq_len=47,
        batch_size=64,
        train_size=65536,
        test_size=65536,
        reverse=False,
        # TODO: Why does `shuffle=True` improve performance so much?
        shuffle=True,
    )
    # train the model with the `optimize_model` function
    model, log = optimize_model(calcium_data, model, num_epochs=10, **kwargs)
    # make predictions with trained model
    targets, predictions = model_predict(model, calcium_data)
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")
    # figure of neuron 0 calcium target and prediction
    plt.figure()
    plt.plot(targets[:, 0], label="target")
    plt.plot(predictions[:, 0], alpha=0.8, label="prediction")
    plt.legend()
    plt.title("Neuron 0 target and prediction")
    plt.xlabel("Time")
    plt.ylabel("$Ca^{2+} \Delta F / F$")
    plt.show()
