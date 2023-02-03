from data.load_neural_activity import pick_worm, load_dataset
from visualization.plot_loss_log import plot_loss_log
from visualization.plot_before_after_weights import plot_before_after_weights
from visualization.plot_correlation_scatter import plot_correlation_scatter
from visualization.plot_target_prediction import plot_target_prediction
from models.linear_models import LinearNN
from train.train_main import optimize_model
from train.train_main import model_predict
import hydra
import numpy as np


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """
    Trains a simple linear regression model.
    config params:
        dataset.name: Name of a dataset of worms.

    """
    # choose your dataset
    dataset_name = config.dataset.name
    all_worms_dataset = load_dataset(dataset_name)
    print("Chosen dataset:", dataset_name, end="\n\n")
    # get data for one worm
    worm = np.random.choice(list(all_worms_dataset.keys()))
    single_worm_dataset = pick_worm(all_worms_dataset, worm)
    print("Picked:", worm, end="\n\n")
    # get the calcium data for this worm
    calcium_data = single_worm_dataset["all_data"]
    print("worm calcium-signal dataset:", calcium_data.shape, end="\n\n")
    # get the neuron ids and length of the recording
    neuron_to_idx = single_worm_dataset["all_neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    neuron_idx = np.random.choice(list(neuron_to_idx.values()))
    max_time = single_worm_dataset["max_time"]
    num_neurons = single_worm_dataset["num_neurons"]
    # create the model and place on GPU
    lin_model = LinearNN(input_size=num_neurons).double()
    # optimize the model (use default settings)
    lin_model, log = optimize_model(
        dataset=calcium_data, model=lin_model, seq_len=range(1, 14), num_epochs=200
    )
    # plot loss curves
    plot_loss_log(
        log,
        plt_title="%s, %s neurons, data size %s, seq. len %s \n "
        "Linear Model: $\Delta Y(t) = W^{\intercal} Y(t)$ loss "
        "curves" % (worm.upper(), num_neurons, log["data_size"], log["seq_len"]),
    )
    # make predictions with the trained linear model.
    neuron = idx_to_neuron[neuron_idx]
    targets, predictions = model_predict(calcium_data, lin_model)
    # plot prediction for a single neuron
    plot_target_prediction(
        targets[:, neuron_idx],
        predictions[:, neuron_idx],
        plt_title="%s, neuron %s, data size %s, seq. len %s \n "
        "Linear model: Ca2+ residuals prediction"
        % (worm.upper(), neuron, log["data_size"], log["seq_len"]),
    )
    # plot scatterplot of all neuron predictions
    plot_correlation_scatter(
        targets,
        predictions,
        plt_title="%s, %s neurons, data size %s, \n Linear "
        "Model: Correlation of all neuron Ca2+ residuals"
        % (worm.upper(), num_neurons, log["data_size"]),
    )
    return None


if __name__ == "__main__":
    main()
