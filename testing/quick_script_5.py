"""
Tests the training pipeline using `train_model`.
"""

from omegaconf import OmegaConf
from models._utils import NetworkLSTM
from train._main import train_model
from data._main import get_dataset
from visualization._utils import plot_targets_predictions

data_config = OmegaConf.load("conf/dataset.yaml")
train_config = OmegaConf.load("conf/train.yaml")

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = get_dataset(data_config)
    # create a model
    model = NetworkLSTM(302, 64).double()
    # run the full train pipeline
    model, log_dir = train_model(model, dataset, train_config)
    # compare predictions against targets
    plot_targets_predictions(log_dir, worm="worm0", neuron="all")
