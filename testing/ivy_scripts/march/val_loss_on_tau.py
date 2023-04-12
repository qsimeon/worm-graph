#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: train_main.py
@time: 2023/3/24 10:15

visualize the relationship between val_loss - baseline and tau_train, and 
quantitative meature two different criteria for "baseline":
(1).    use y_t(current) as the prediction
(2).    use the average of data[t + tau: t + tau + seq_len]
"""

from train._utils import *


def train_model(
    model: torch.nn.Module,
    dataset: dict,
    config: DictConfig,
    shuffle: bool = True,  # whether to shuffle worms
) -> tuple[torch.nn.Module, str]:
    """
    Trains a model on a multi-worm dataset. Returns the trained model
    and a path to the directory with training and evaluation logs.
    """
    assert "worm0" in dataset, "Not a valid dataset object."
    # initialize
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print("--------timestamp---------")
    print(timestamp)
    log_dir = os.path.join(
        "logs", "{}-{}-{}".format(timestamp, dataset_name, model_class_name)
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    # sample worms with replacement until desired number epochs (i.e. worms) obtained
    dataset_items = [
        (k, dataset[k])
        for k in np.random.choice(
            list(dataset.keys()), size=config.train.epochs, replace=True
        )
    ]
    # shuffle the worms in dataset (without replacement)
    if shuffle == True:
        dataset_items = random.sample(dataset_items, k=len(dataset_items))
    # remake dataset with only selected worms
    dataset = dict(dataset_items)
    # instantiate the optimizer
    opt_param = config.train.optimizer
    learn_rate = config.train.learn_rate
    if config.train.optimizer is not None:
        if isinstance(opt_param, str):
            optimizer = eval(
                "torch.optim."
                + opt_param
                + "(model.parameters(), lr="
                + str(learn_rate)
                + ")"
            )
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), "Please use an instance of torch.optim.Optimizer."
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    # print("Optimizer:", optimizer, end="\n\n")
    # get other config params
    if config.get("globals"):
        use_residual = config.globals.use_residual
        smooth_data = config.train.smooth_data
    else:
        use_residual = False
        smooth_data = False
    # train/test loss metrics
    data = {
        "epochs": [],
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
        "num_train_samples": [],
        "num_test_samples": [],
        "centered_train_losses": [],
        "centered_test_losses": [],
    }
    # train the model for multiple cyles
    kwargs = dict(  # args to `split_train_test`
        k_splits=config.train.k_splits,
        seq_len=config.train.seq_len,
        batch_size=config.train.batch_size,
        train_size=config.train.train_size,
        test_size=config.train.test_size,
        shuffle=config.train.shuffle,  # whether to shuffle the samples from a worm
        reverse=False,
        tau=config.train.tau_in,
        use_residual=use_residual,
    )
    # choose whether to use calcium or residual data
    if use_residual:
        key_data = "residual_calcium"
    else:
        key_data = "calcium_data"
    # choose whether to use original or smoothed data
    if smooth_data:
        key_data = "smooth_" + key_data
    else:
        key_data = key_data
    # memoize creation of data loaders and masks for speedup
    memo_loaders_masks = dict()
    # train for config.train.num_epochs
    reset_epoch = 1
    for i, (worm, single_worm_dataset) in enumerate(dataset_items):
        # check memo for loaders and masks
        if worm in memo_loaders_masks:
            train_loader = memo_loaders_masks[worm]["train_loader"]
            test_loader = memo_loaders_masks[worm]["test_loader"]
            train_mask = memo_loaders_masks[worm]["train_mask"]
            test_mask = memo_loaders_masks[worm]["test_mask"]
        else:
            # create data loaders and train/test masks only once per worm
            train_loader, test_loader, train_mask, test_mask = split_train_test(
                data=single_worm_dataset[key_data],
                time_vec=single_worm_dataset.get(
                    "time_in_seconds", None
                ),  # time vector
                **kwargs,
            )
            # add to memo
            memo_loaders_masks.setdefault(
                worm,
                dict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    train_mask=train_mask,
                    test_mask=test_mask,
                ),
            )
        # mutate the dataset for this worm with the train and test masks
        dataset[worm].setdefault("train_mask", train_mask)
        dataset[worm].setdefault("test_mask", test_mask)
        # get the neurons mask for this worm
        neurons_mask = single_worm_dataset["named_neurons_mask"]
        # optimize for 1 epoch per (possibly duplicated) worm
        model, log = optimize_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            neurons_mask=neurons_mask,
            optimizer=optimizer,
            start_epoch=reset_epoch,
            learn_rate=learn_rate,
            num_epochs=1,
            use_residual=use_residual,
        )
        # retrieve losses and sample counts
        [data[key].extend(log[key]) for key in data]  # Python list comprehension
        # set to next epoch
        reset_epoch = log["epochs"][-1] + 1
        # outputs
        if (i % config.train.save_freq == 0) or (i + 1 == config.train.epochs):
            # display progress
            print("num. worms trained on:", i + 1, "\nprevious worm:", worm, end="\n\n")
            # save model checkpoints
            chkpt_name = "{}_epochs_{}_worms.pt".format(reset_epoch - 1, i + 1)
            torch.save(
                {
                    "epoch": reset_epoch - 1,
                    "model_name": model_class_name,
                    "input_size": model.get_input_size(),
                    "hidden_size": model.get_hidden_size(),
                    "num_worms": i + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(log_dir, "checkpoints", chkpt_name),
            )
    # save loss curves
    pd.DataFrame(data=data).to_csv(
        os.path.join(log_dir, "loss_curves.csv"),
        index=True,
        header=True,
    )
    # make predictions with last saved model
    make_predictions(
        model,
        dataset,
        log_dir,
        tau=config.train.tau_out,
        use_residual=use_residual,
        smooth_data=smooth_data,
    )
    # returned trained model and a path to log directory
    return model, log_dir


if __name__ == "__main__":
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")

    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))

    model_config = OmegaConf.load("conf/model.yaml")
    # model_config.model.input_size = dataset["worm0"]["named_neurons_mask"].shape[0]
    model = get_model(model_config)

    # if not all worms are needed, e.g. here we only choose one worm
    for i in range(len(dataset) - 1):
        d = dataset.popitem()

    tau_range = range(1, config.train.seq_len * 10 + 5, config.train.seq_len)
    val_loss = []
    ori_val_loss = []

    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]

    path_par = os.getcwd()
    path_par += "/testing/ivy_scripts/figure"
    isExist = os.path.exists(path_par)
    if not isExist:
        os.mkdir(path_par)

    path = os.getcwd()
    path += "/testing/ivy_scripts/figure/seq_len=" + str(config.train.seq_len)
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)

    for t in tau_range:
        config.train.tau_in = t
        config.train.tau_out = t
        model, log_dir = train_model(model, dataset, config)
        loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)
        val_loss.append(loss_df["centered_test_losses"].get(config.train.epochs - 1))
        ori_val_loss.append(loss_df["test_losses"].get(config.train.epochs - 1))

        targets, predictions = model_predict(model, calcium_data)
        print(
            "Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n"
        )

        for neuron in [12, 22]:
            plt.figure()
            plt.plot(targets[:, neuron], label="target")
            plt.plot(
                range(t, t + predictions[:, neuron].shape[0]),
                predictions[:, neuron],
                alpha=0.8,
                label="prediction",
            )
            plt.legend()
            plt.title(
                "Neuron "
                + str(neuron)
                + " target and prediction on tau = "
                + str(t)
                + "\n baseline = current"
            )
            plt.xlabel("Time")
            plt.ylabel("$Ca^{2+} \Delta F / F$")
            plt.savefig(
                os.path.join(
                    path,
                    "Neuron "
                    + str(neuron)
                    + " target and prediction on tau = "
                    + str(t)
                    + ".png",
                )
            )

    plt.figure()
    plt.plot(tau_range, val_loss)
    plt.plot(tau_range, ori_val_loss)
    plt.legend(["cen_loss", "ori_loss"], loc="upper right")
    plt.ylabel("MSE loss")
    plt.xlabel("tau (tau_in == tau_out == tau)")
    plt.title(
        "val_loss - baseline on tau \n baseline: current  worm: worm0  dataset: Uzel2022"
    )
    plt.show()
