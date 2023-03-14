import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import LinearNN
from train._main import train_model
from data._main import get_dataset
from train._utils import optimize_model
from visualization._utils import plot_before_after_weights

if __name__ == "__main__":
    # load dataset and get calcium data for one worm
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    calcium_data = dataset["worm0"]["calcium_data"]
    # create a model
    model = LinearNN(302, 64).double()
    # train the model
    kwargs = dict(train_size=4096, test_size=4096, tau=1, seq_len=47, reverse=False)
    model, log = optimize_model(calcium_data, model, k_splits=2, **kwargs)

    plt.figure()

    plt.plot(log["train_mask"].to(float).numpy())

    plt.title("Train mask")

    plt.xlabel("Time")

    plt.ylabel("Test (0) / Train (1)")

    plt.show()

    config = OmegaConf.load("conf/train.yaml")

    model, log_dir = train_model(model, dataset, config, shuffle=True)

    plot_before_after_weights(log_dir)
