from train._utils import *


def train_model(model: torch.nn.Module, dataset, config: DictConfig):
    optim = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
    logs = dict()
    for worm, single_worm_dataset in dataset:
        model, log = optimize_model(
            dataset=single_worm_dataset["calcium_data"],
            model=model,
            mask=single_worm_dataset["named_neurons_mask"],
            optimizer=optim,
            num_epochs=config.epochs,
            seq_len=config.seq_len,
            dataset_size=config.dataset_size,
        )
        logs[worm] = log
    return model, log


if __name__ == "__main__":
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    config = OmegaConf.load("conf/train.yaml")
    train_model(model, dataset, config)
