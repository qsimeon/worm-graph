#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: val_loss_vs_train_size.py
@time: 2023/3/30 15:25
"""

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
    single_size = config.train.train_size
    train_size_range = range(10, single_size, 50)

    datasets = ["Uzel2022", "Flavell2023", "Kato2015"]

    for d in datasets:
        time_range = []
        hyper_log = []

        model = get_model(OmegaConf.load("conf/model.yaml"))
        config_data = OmegaConf.load("conf/dataset.yaml")
        config_data.dataset.name = d
        dataset = get_dataset(config_data)

        for train_size in train_size_range:
            config.train.train_size = train_size

            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            time_range.append(timestamp)

            model, log_dir = train_model(
                model,
                dataset,
                config,
                shuffle=config.train.shuffle,
                log_dir=os.path.join("logs", "{}".format(timestamp)),
            )

            loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)

            val_loss = np.array(loss_df["centered_test_losses"])[-len(dataset) :].mean()
            hyper_log.append(val_loss)
        plt.plot(train_size_range, hyper_log)

    plt.ylabel("validation loss - baseline")
    plt.xlabel("train_size")
    plt.title("model = LSTM(_, 64, 1), seq_len=tau=100, epoch=100")
    plt.legend(datasets)
    plt.savefig(
        os.path.join(
            os.getcwd() + "/testing/ivy_scripts/figures/",
            "train_size_small_adam_avg.png",
        )
    )
    plt.show()
