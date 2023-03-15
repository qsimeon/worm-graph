"""
Tests the full training pipeline function `train_model`.
"""

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import LinearNN
from train._main import train_model
from data._main import get_dataset
from visualization._utils import plot_targets_predictions

data_config = OmegaConf.load("conf/dataset.yaml")
train_config = OmegaConf.load("conf/train.yaml")

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = get_dataset(data_config)
    # create a model
    model = LinearNN(302, 64).double()
    # run the full train pipeline
    model, log_dir = train_model(model, dataset, train_config)
    # compare predictions against targets
    plot_targets_predictions(log_dir, worm="worm5", neuron="all")
