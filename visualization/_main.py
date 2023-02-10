from visualization._utils import *


def plot_figures(config: DictConfig) -> None:
    plot_loss_log(
        log,
        plt_title="%s, %s neurons, data size %s, seq. len %s \n "
        "Linear Model: $\Delta Y(t) = W^{\intercal} Y(t)$ loss "
        "curves"
        % (dataset["worm"].upper(), num_neurons, log["data_size"], log["seq_len"]),
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
    plot_figures(OmegaConf.load("conf/visualize.yaml"))
