from train._utils import *

# Init logger
logger = logging.getLogger(__name__)


def train_model(
    train_config: DictConfig,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    verbose: bool = False,
):
    """
    Main standard ML training loop.

    Parameters
    ----------
    train_config : DictConfig
        A dictionary containing the training configuration.
    model : torch.nn.Module
        A model instance.
    train_dataset : torch.utils.data.Dataset
        A train dataset instance.
    val_dataset : torch.utils.data.Dataset
        A validation dataset instance.
    verbose : bool, optional
        Whether to print statistics at each epoch, by default False (True when using MULTIRUN hydra mode)

    Returns
    -------
    model: torch.nn.Module
        The trained model.
    metric: float
        The metric used for optuna (lowest validation loss across epochs).
    """

    # Verifications
    assert (
        isinstance(train_config.epochs, int) and train_config.epochs > 0
    ), "epochs must be a positive integer"
    assert (
        isinstance(train_config.batch_size, int) and train_config.batch_size > 0
    ), "batch_size must be a positive integer"
    assert isinstance(train_config.shuffle, bool), "shuffle must be a boolean"
    assert train_config.optimizer in [
        "SGD",
        "Adam",
        "AdamW",
        "Adagrad",
        "Adadelta",
        "RMSprop",
    ], "optimizer must be one of SGD, Adam, AdamW, Adagrad, Adadelta, RMSprop"

    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}

    # Create train and checkpoints directories
    os.makedirs(os.path.join(log_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "train", "checkpoints"), exist_ok=True)

    # Load model to device
    model = model.to(DEVICE)

    # Parameters
    epochs = train_config.epochs
    batch_size = train_config.batch_size
    shuffle = train_config.shuffle
    criterion = model.loss_fn()
    save_freq = train_config.save_freq

    # Optimizer parameters
    optim_name = "torch.optim." + train_config.optimizer
    lr = train_config.lr
    optimizer = eval(optim_name + "(model.parameters(), lr=" + str(lr) + ")")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # Instantiate EarlyStopping
    es = EarlyStopping(
        patience=train_config.early_stopping.patience,
        min_delta=train_config.early_stopping.delta,
    )

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Loss metrics
    train_running_base_loss = 0
    train_running_loss = 0
    train_epoch_loss = []
    train_epoch_baseline = []

    val_running_base_loss = 0
    val_running_loss = 0
    val_epoch_loss = []
    val_epoch_baseline = []

    # Computation metrics
    computation_time = []
    computation_flops = []
    num_trainable_params = []

    # Start training
    logger.info("Starting training loop...")

    pbar = tqdm(
        range(epochs + 1),  # +1 because we skip the first epoch
        total=epochs + 1,
        position=0,
        leave=True,  # position at top and remove when done
        dynamic_ncols=True,  # adjust width to terminal window size
    )

    for epoch in pbar:
        # ============================ Train loop ============================

        model.train()

        # Measure the time for an epoch
        start_time = time.perf_counter()

        for batch_idx, (X_train, Y_train, masks_train, metadata_train) in enumerate(
            trainloader
        ):
            X_train = X_train.to(DEVICE)
            Y_train = Y_train.to(DEVICE)
            masks_train = masks_train.to(DEVICE)

            optimizer.zero_grad()  # Reset gradients

            # Baseline model: identity model - predict that the next time step is the same as the current one.
            # This is the naive predictor: predicting that the next time step is the same as the current one is
            # better in expectation than predicting a random value.
            y_base = X_train
            train_baseline = compute_loss_vectorized(
                loss_fn=criterion, X=y_base, Y=Y_train, masks=masks_train
            )

            # Models operate sequence-to-sequence.
            y_pred = model(X_train, masks_train)
            train_loss = compute_loss_vectorized(
                loss_fn=criterion, X=y_pred, Y=Y_train, masks=masks_train
            )

            # Backpropagation (skip first epoch)
            if epoch > 0:
                train_loss.backward()
                optimizer.step()

            # Update running losses
            train_running_base_loss += train_baseline.item()
            train_running_loss += train_loss.item()

        # Compute the time taken
        end_time = time.perf_counter()
        computation_time.append(end_time - start_time)

        # Store metrics
        train_epoch_loss.append(train_running_loss / len(trainloader))
        train_epoch_baseline.append(train_running_base_loss / len(trainloader))

        # Reset running losses
        train_running_base_loss = 0
        train_running_loss = 0

        # ============================ Validation loop ============================

        model.eval()

        with torch.no_grad():
            for batch_idx, (X_val, Y_val, masks_val, metadata_val) in enumerate(
                valloader
            ):
                X_val = X_val.to(DEVICE)
                Y_val = Y_val.to(DEVICE)
                masks_val = masks_val.to(DEVICE)

                # Baseline model: identity model - predict that the next time step is the same as the current one.
                # This is the simplest model we can think of: predict that the next time step is the same as the current one
                # is better than predict any other random number.
                y_base = X_val
                val_baseline = compute_loss_vectorized(
                    loss_fn=criterion, X=y_base, Y=Y_val, masks=masks_val
                )

                # Models operate sequence-to-sequence.
                y_pred = model(X_val, masks_val)
                val_loss = compute_loss_vectorized(
                    loss_fn=criterion, X=y_pred, Y=Y_val, masks=masks_val
                )

                # Update running losses
                val_running_base_loss += val_baseline.item()
                val_running_loss += val_loss.item()

            # Compute FLOPs and number of trainable parametes
            if epoch == 0:
                flops = FlopCountAnalysis(model, (X_train, masks_train))
                param_ct = sum(p.numel() for p in model.parameters() if p.requires_grad)
            computation_flops.append(flops.total())
            num_trainable_params.append(param_ct)

            # Store metrics
            val_epoch_loss.append(val_running_loss / len(valloader))
            val_epoch_baseline.append(val_running_base_loss / len(valloader))

            # Reset running losses
            val_running_base_loss = 0
            val_running_loss = 0

        # Step the scheduler
        scheduler.step(val_epoch_loss[-1])
        current_lr = optimizer.param_groups[0]["lr"]

        # Save model checkpoint
        if epoch % save_freq == 0:
            save_model(
                model,
                os.path.join(
                    log_dir, "train", "checkpoints", "model_epoch_" + str(epoch) + ".pt"
                ),
            )

        # Early stopping
        if es(model, val_epoch_loss[-1]):  # on validation loss
            logger.info("Early stopping triggered (epoch {}).".format(epoch))
            break

        # Print statistics if in verbose mode
        if verbose:
            logger.info(
                "Epoch: {}/{} | Train loss: {:.4f} | Val. loss: {:.4f}".format(
                    epoch, epochs + 1, train_epoch_loss[-1], val_epoch_loss[-1]
                )
            )

        # Update progress bar
        pbar.set_description(f"Epoch {epoch}/{epochs}")
        pbar.set_postfix(
            {"Train loss": train_epoch_loss[-1], "Val. loss": val_epoch_loss[-1]}
        )

    # Restore best model and save it
    logger.info("Training loop is over. Loading best model.")
    model.load_state_dict(es.best_model.state_dict())
    save_model(model, os.path.join(log_dir, "train", "checkpoints", "model_best.pt"))

    # Save training and evaluation metrics into a csv file
    train_metrics = pd.DataFrame(
        {
            "epoch": np.arange(len(train_epoch_loss)),
            "computation_time": computation_time,
            "computation_flops": computation_flops,
            "num_trainable_params": num_trainable_params,
            "learning_rate": current_lr,
            "train_loss": train_epoch_loss,
            "train_baseline": train_epoch_baseline,
            "val_loss": val_epoch_loss,
            "val_baseline": val_epoch_baseline,
        }
    )
    train_metrics.to_csv(
        os.path.join(log_dir, "train", "train_metrics.csv"), index=False
    )

    # Metric for optuna (lowest validation loss)
    metric = min(val_epoch_loss)
    if metric == np.NaN or metric is None:
        metric = float("inf")

    return model, metric


if __name__ == "__main__":
    train_config = OmegaConf.load("configs/submodule/train.yaml")
    print(OmegaConf.to_yaml(train_config), end="\n\n")

    model_config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(model_config), end="\n\n")

    dataset_config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(dataset_config), end="\n\n")

    # Create new to log directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("logs/hydra", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Move to new log directory
    os.chdir(log_dir)

    # Dataset
    train_dataset, val_dataset = get_datasets(dataset_config.dataset)

    # Get the model
    model = get_model(model_config.model)

    # Train the model
    model, metric = train_model(
        train_config=train_config.train,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    print("Final metric:", metric)
