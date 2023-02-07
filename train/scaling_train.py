import torch
import numpy as np
from visualization.plot_loss_log import plot_loss_log
from visualization.plot_target_prediction import plot_target_prediction
from visualization.plot_correlation_scatter import plot_correlation_scatter
from train.train_main import model_predict, optimize_model
from data.load_neural_activity import pick_worm

# @title Functions for training on more data and multiple worms.
# @markdown


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
    A function to investigate the effect of the amount of data
    (sequence length fixed) on the training and generalization of a model.
    """
    results_list = []
    # parse worm dataset
    # get the calcium data for this worm
    new_calcium_data = single_worm_dataset["named_data"]
    mask = single_worm_dataset["neurons_mask"]
    # get the neuron to idx map
    neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # pick a neuron
    neuron_idx = np.random.choice(list(neuron_to_idx.keys())) - 1
    neuron = neuron_to_idx[neuron_idx]
    # get max time and number of neurons
    max_time = single_worm_dataset["max_time"]
    num_neurons = single_worm_dataset["num_neurons"]
    # iterate over different dataset sizes
    data_sizes = np.logspace(5, np.floor(np.log2(max_time // 2)), 10, base=2, dtype=int)
    print("Training dataset sizes we will try:", data_sizes.tolist())
    for data_size in data_sizes:
        print()
        print("Dataset size", data_size)
        # initialize model
        model = model_class(input_size=302).double()
        # train the model on this amount of data
        model, log = optimize_model(
            new_calcium_data,
            model,
            mask,
            num_epochs=num_epochs,
            seq_len=seq_len,
            data_size=data_size,
        )
        # put the worm and neuron in log
        log["worm"] = worm
        log["neuron_idx"] = neuron_idx
        log["neuron"] = neuron
        log["num_neurons"] = num_neurons
        # true dataset size
        size = log["data_size"]
        # predict with the model
        targets, predictions = model_predict(new_calcium_data, model)
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
                % (worm.upper(), neuron, size, seq_len, model_name),
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


def leave_one_worm_out_training(
    model_class,
    multi_worms_dataset,
    num_epochs=100,
    model_name="",
    seq_len=1,
    plotting=True,
):
    """
    Train on all but one worm in a dataset
    and test on that one worm.
    """
    leaveOut_worm = np.random.choice(list(multi_worms_dataset))
    test_worm_dataset = multi_worms_dataset[leaveOut_worm]
    train_worms_dataset = {
        worm: dataset
        for worm, dataset in multi_worms_dataset.items()
        if worm != leaveOut_worm
    }
    # initialize the model
    model = model_class(input_size=302).double()
    # train the model sequentially on many worms
    train_results = multi_worm_training(
        model_class,
        train_worms_dataset,
        num_epochs,
        model_name,
        seq_len,
        plotting=False,
    )
    model, train_log = train_results[-1]
    # get the calcium data for left out worm
    new_calcium_data = test_worm_dataset["named_data"]
    mask = test_worm_dataset["neurons_mask"]
    # get the neuron to idx map
    neuron_to_idx = test_worm_dataset["named_neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # pick a neuron
    neuron_idx = np.random.choice(list(neuron_to_idx.keys())) - 1
    neuron = neuron_to_idx[neuron_idx]
    # get max time and number of neurons
    max_time = test_worm_dataset["max_time"]
    num_neurons = test_worm_dataset["num_neurons"]
    # predict with the model
    targets, predictions = model_predict(new_calcium_data, model)
    # create a log for this evaluation
    log = dict()
    log.update(train_log)
    log["worm"] = leaveOut_worm
    log["neuron_idx"] = neuron_idx
    log["neuron"] = neuron
    log["num_neurons"] = num_neurons
    log["targets"] = targets
    log["predictions"] = predictions
    size = train_log["data_size"]
    # test the model on the left out worm
    if plotting:
        # plot final loss curve
        plot_loss_log(
            log,
            plt_title="%s, data size %s, seq. len %s "
            "\n %s Model: Loss curves"
            % (set(w.upper() for w in train_worms_dataset), size, seq_len, model_name),
        )
        # plot prediction for a single neuron
        plot_target_prediction(
            targets[:, neuron_idx],
            predictions[:, neuron_idx],
            plt_title="%s, neuron %s, data size %s, seq. len "
            "%s \n %s Model: Ca2+ residuals prediction"
            % (leaveOut_worm.upper(), neuron, size, seq_len, model_name),
        )
        # plot scatterplot of all neuron predictions
        plot_correlation_scatter(
            targets[:, mask],
            predictions[:, mask],
            plt_title="%s, %s neurons,"
            " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
            "residuals"
            % (leaveOut_worm.upper(), num_neurons, size, seq_len, model_name),
        )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return log


def multi_worm_training(
    model_class,
    multi_worms_dataset,
    num_epochs=100,
    model_name="",
    seq_len=1,
    plotting=False,
):
    """
    A helper function to investigate the effect of training a model
    on increasingly more worms.
    """
    print("Number of worms in this dataset:", len(multi_worms_dataset))
    results_list = []
    worms_seen = set()
    for worm in multi_worms_dataset:
        print()
        print("Trained so far on", sorted(worms_seen))
        print("Currently training on", worm)
        worms_seen.add(worm)
        # parse worm dataset
        single_worm_dataset = pick_worm(multi_worms_dataset, worm)
        # get the calcium data for this worm
        new_calcium_data = single_worm_dataset["named_data"]
        mask = single_worm_dataset["neurons_mask"]
        # get the neuron to idx map
        neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
        idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
        # pick a neuron
        neuron_idx = np.random.choice(list(neuron_to_idx.keys())) - 1
        neuron = neuron_to_idx[neuron_idx]
        # get max time and number of neurons
        max_time = single_worm_dataset["max_time"]
        num_neurons = single_worm_dataset["num_neurons"]
        # initialize model and an optimizer
        model = model_class(input_size=302).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # train the model on this worm's  data
        model, log = optimize_model(
            new_calcium_data,
            model,
            mask,
            optimizer,
            data_size=2048,
            num_epochs=num_epochs,
            seq_len=seq_len,
        )
        # put the worm and neuron in log
        log["worm"] = worm
        log["neuron_idx"] = neuron_idx
        log["neuron"] = neuron
        log["num_neurons"] = num_neurons
        # true dataset size
        size = log["data_size"]
        # predict with the model
        targets, predictions = model_predict(new_calcium_data, model)
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
                % (worm.upper(), neuron, size, seq_len, model_name),
            )
            # plot scatterplot of all neuron predictions
            plot_correlation_scatter(
                targets[:, mask],
                predictions[:, mask],
                plt_title="%s, %s neurons,"
                " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
                "residuals" % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # add to results
        results_list.append((model, log))
    return results_list
