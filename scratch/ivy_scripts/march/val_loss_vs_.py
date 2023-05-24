#!/usr/bin/env python
# encoding: utf-8

from train._main import *
import matplotlib.pyplot as plt


def specific_log(x, y, timestamp, log_dir):
    hyper_log = dict()
    loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)
    hyper_log.setdefault(
        timestamp,
        {
            str(x): x,
            str(y): loss_df[str(y)][-1],
        },
    )
    return hyper_log


if __name__ == "__main__":
    config = OmegaConf.load("conf/train.yaml")
    print(config.train.epochs)

    datasets = ["Uzel2022", "Flavell2023", "Nichols2017", "Kota2015"]

    for d in datasets:
        time_range = []
        hyper_log = []

        model = get_model(OmegaConf.load("conf/model.yaml"))
        config_data = OmegaConf.load("conf/dataset.yaml")
        config_data.dataset.name = d
        dataset = get_dataset(config_data)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        model, log_dir = train_model(
            model,
            dataset,
            config,
            shuffle=config.train.shuffle,
            log_dir=os.path.join("logs", "{}".format(timestamp)),
        )

        loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)
        plt.plot(range(config.train.epochs), loss_df["centered_test_losses"])

    plt.ylabel("validation loss - baseline")
    plt.xlabel("epoch")
    plt.legend(datasets)
    plt.savefig(
        os.path.join(
            os.getcwd() + "/testing/ivy_scripts/figures/", "epoch_datasets.png"
        )
    )
    plt.show()
