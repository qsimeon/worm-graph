import torch
from torch_geometric.data import Data
import os

from utils import ROOT_DIR

from data._main import *
from train._main import *

if __name__ == "__main__":
    # leave one worm for training
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    for i in range(len(dataset) - 1):
        d = dataset.popitem()

    # list of possible taus
    tau_range = list(range(1, 30, 3))
    r2 = list(range(30, 301, 30))
    tau_range.extend(r2)

    # go on training, only the val. loss of the last epoch will be recorded
    val_loss = []
    ori_val_loss = []
    for tau in tau_range:
        config = OmegaConf.load("conf/train.yaml")
        config.train.tau_in = tau
        print("config:", OmegaConf.to_yaml(config), end="\n\n")
        model = get_model(OmegaConf.load("conf/model.yaml"))
        model, log_dir = train_model(model, dataset, config)
        loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)
        val_loss.append(loss_df["centered_test_losses"].get(config.train.epochs - 1))
        ori_val_loss.append(loss_df["test_losses"].get(config.train.epochs - 1))

    plt.plot(tau_range, val_loss)
    plt.plot(tau_range, ori_val_loss)
    plt.legend(["cen_loss", "ori_loss"], loc="upper right")
    plt.ylabel("MSE loss")
    plt.xlabel("tau")
    plt.title("val_loss - baseline on tau \n baseline: current  worm: worm0  dataset: Uzel2022")
    plt.show()
