"""
Tests the model optimization, the full training 
pipeline and the train test masks.
"""
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import LinearNN
from train._main import train_model
from data._main import get_dataset
from train._utils import optimize_model
from visualization._utils import plot_before_after_weights

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # load a dataset (multiple worms)
    dataset = get_dataset(config)
    # get calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = dataset["calcium_data"]
    # create a model
    model = LinearNN(302, 64).double()
    # test the  `optimize_model` function
    kwargs = dict(train_size=4096, test_size=4096, tau=1, seq_len=47, reverse=False)
    model, log = optimize_model(calcium_data, model, k_splits=2, **kwargs)
    # run the full train pipeline
    config = OmegaConf.load("conf/train.yaml")
    model, log_dir = train_model(model, dataset, config, shuffle=True)
    # plot figure showing train mask
    plt.figure()
    plt.plot(log["train_mask"].to(float).numpy())
    plt.title("Train mask")
    plt.xlabel("Time")
    plt.ylabel("Test (0) / Train (1)")
    plt.show()
    # plot untrained versus trained weights
    plot_before_after_weights(log_dir)
