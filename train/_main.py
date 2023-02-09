from train._utils import *


def train_model(model: torch.nn.Module, dataset: dict, config: DictConfig):
    for single_worm_dataset in dataset:
        model, log =  optimize_model(dataset=dataset["calcium_data"], model=model, seq_len=7, num_epochs=config.epochs)
    return None


if __name__ == "__main__":
    train_model(OmegaConf.load("conf/train.yaml"))
