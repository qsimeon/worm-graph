"""
Tests the training pipeline using 
the function `train_model`.
"""

from omegaconf import OmegaConf
from utils import NEURONS_302
from models._utils import NetworkLSTM
from train._main import train_model
from data._utils import load_Uzel2022
from visualization._utils import plot_loss_curves, plot_targets_predictions

if __name__ == "__main__":
    # specify the training config
    config = OmegaConf.create(
        dict(
            train=dict(
                learn_rate=0.01,
                seq_len=69,
                k_splits=2,
                epochs=100,
                save_freq=100,
                smooth_data=False,
                batch_size=128,
                train_size=1654,
                test_size=1654,
                shuffle=True,  # whether to shuffle sample
                tau_in=1,
                tau_out=1,
                optimizer="SGD",
            )
        )
    )
    print(OmegaConf.to_yaml(config))
    # load a dataset (multiple worms)
    dataset = load_Uzel2022()
    # create a model
    model = NetworkLSTM(302, 64).double()
    # run the full train pipeline
    model, log_dir = train_model(model, dataset, config)
    # plot the loss curves
    plot_loss_curves(log_dir)
    # compare predictions against targets
    for i in range(50):
        neuron = NEURONS_302[i]
        plot_targets_predictions(
            log_dir,
            worm="worm0",
            neuron=neuron,
        )
