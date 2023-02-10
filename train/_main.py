from train._utils import *


def train_model(model: torch.nn.Module, dataset, config: DictConfig):
    # initialize
    logs = dict()
    epoch = 1

    # train the model
    optim = torch.optim.Adam(model.parameters(), lr=config.train.learn_rate)
    for worm, single_worm_dataset in dataset:
        model, log = optimize_model(
            dataset=single_worm_dataset["calcium_data"],
            model=model,
            mask=single_worm_dataset["named_neurons_mask"],
            optimizer=optim,
            start_epoch=epoch,
            num_epochs=config.train.epochs,
            seq_len=config.train.seq_len,
            dataset_size=config.train.dataset_size,
        )
        logs[worm] = log
        epoch = log["epochs"][-1] + 1

    # save trained model weights and optimizer as checkpoint
    save_path = os.path.join(
        config.train.log_dir,
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pt",
    )
    torch.save(
        {
            "epoch": log["epochs"][-1],
            "train_loss": log["train_losses"][-1],
            "val_loss": log["test_losses"][-1],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        save_path,
    )

    # make predicitons with trained model? (and add to logs)
    for worm, single_worm_dataset in dataset:
        targets, predictions = model_predict(single_worm_dataset["calcium_data"], model)
        logs[worm].update(
            {
                "target_calcium_residual": targets,
                "predicted_calcium_residual": predictions,
            }
        )
        print(logs[worm], end="\n\n")
        
    # save logs as Pandas dataframe(s)
    print(logs, end="\n\n")
    return model, logs


if __name__ == "__main__":
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model, logs = train_model(model, dataset, config)
    print("logs:", logs, end="\n\n")
