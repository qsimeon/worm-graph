import numpy as np
import model_predict
from visualization import (
    plot_loss_log,
    plot_target_prediction,
    plot_correlation_scatter,
)
from train_main import optimize_model


# @title Functions for training on more data and multiple worms.


def more_data_training(
    model_class,
    single_worm_dataset,
    num_epochs=100,
    worm="worm**",
    model_name="",
    seq_len=1,
    plotting=False,
):
    """
    A function to investigate the effect of training models more data
    (fixed sequence length).
    """
    results_list = []
    # parse worm dataset
    calcium_data = single_worm_dataset["data"]
    neuron_ids = single_worm_dataset["neuron_ids"]
    neuron_idx = np.random.choice(list(neuron_ids.keys())) - 1
    nid = neuron_ids[neuron_idx]
    num_neurons = single_worm_dataset["num_neurons"]
    max_time = single_worm_dataset["max_time"]
    data_sizes = np.logspace(5, np.floor(np.log2(max_time // 2)), 10, base=2, dtype=int)
    print("Training dataset sizes we will try:", data_sizes.tolist())
    for data_size in data_sizes:
        print()
        print("Dataset size", data_size)
        # initialize model
        model = model_class(input_size=num_neurons).double()
        # train the model on this amount of data
        model, log = optimize_model(
            calcium_data,
            model,
            num_epochs=num_epochs,
            seq_len=seq_len,
            data_size=data_size,
        )
        # put the worm and neuron in log
        log["worm"] = worm
        log["neuron_idx"] = neuron_idx
        log["nid"] = nid
        log["num_neurons"] = num_neurons
        # true dataset size
        size = log["data_size"]
        # predict with the model
        targets, predictions = model_predict(single_worm_dataset, model)
        # log targets and predictions
        log["targets"] = targets
        log["predictions"] = predictions
        if plotting:
            # plot loss curves
            plot_loss_log(
                log,
                plt_title="%s, %s neurons, data size %s, seq. len %s "
                "\n %s Model: Loss curves"
                % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
            # plot prediction for a single neuron
            plot_target_prediction(
                targets[:, neuron_idx],
                predictions[:, neuron_idx],
                plt_title="%s, neuron %s, data size %s, seq. len %s"
                " \n %s Model: Ca2+ residuals prediction"
                % (worm.upper(), nid, size, seq_len, model_name),
            )
            # plot scatterplot of all predictions
            plot_correlation_scatter(
                targets,
                predictions,
                plt_title="%s, %s neurons,"
                " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
                "residuals" % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # add to results
        results_list.append((model, log))
    return results_list


def multi_worm_training(
    model_class,
    multi_worms_dataset,
    num_epochs=100,
    model_name="",
    seq_len=1,
    plotting=False,
):
    """
    A helper function to investigate the effect of training a models of
    different worms.
    """
    print("Number of worms in this dataset:", len(multi_worms_dataset))
    results_list = []
    for worm in multi_worms_dataset:
        print()
        print("Currently training on", worm)
        # parse worm dataset
        single_worm_dataset = pick_worm(multi_worms_dataset, worm)
        calcium_data = single_worm_dataset["data"]
        neuron_ids = single_worm_dataset["neuron_ids"]
        neuron_idx = np.random.choice(list(neuron_ids.keys())) - 1
        nid = neuron_ids[neuron_idx]
        num_neurons = single_worm_dataset["num_neurons"]
        # initialize model
        model = model_class(input_size=num_neurons).double()
        # train the model on this worm's  data
        model, log = optimize_model(calcium_data, model, num_epochs=num_epochs)
        # put the worm and neuron in log
        log["worm"] = worm
        log["neuron_idx"] = neuron_idx
        log["nid"] = nid
        log["num_neurons"] = num_neurons
        # true dataset size
        size = log["data_size"]
        # predict with the model
        targets, predictions = model_predict(single_worm_dataset, model)
        # log targets and predictions
        log["targets"] = targets
        log["predictions"] = predictions
        # plot figures
        if plotting:
            # plot loss curves
            plot_loss_log(
                log,
                plt_title="%s, %s neurons, data size %s, seq. len %s "
                "\n %s Model: Loss curves"
                % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
            # plot prediction for a single neuron
            plot_target_prediction(
                targets[:, neuron_idx],
                predictions[:, neuron_idx],
                plt_title="%s, neuron %s, data size %s, seq. len "
                "%s \n %s Model: Ca2+ residuals prediction"
                % (worm.upper(), nid, size, seq_len, model_name),
            )
            # plot scatterplot of all neuron predictions
            plot_correlation_scatter(
                targets,
                predictions,
                plt_title="%s, %s neurons,"
                " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
                "residuals" % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # add to results
        results_list.append((model, log))
    return results_list
