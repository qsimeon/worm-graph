from data._main import get_dataset

from omegaconf import OmegaConf

from train._utils import optimize_model

from train._main import train_model

from models._utils import LinearNN

from visualization._utils import plot_before_after_weights

import matplotlib.pyplot as plt

dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))

calcium_data = dataset["worm0"]["calcium_data"]

model = LinearNN(302, 64).double()

kwargs = dict(train_size=1000, test_size=1000, tau=1, seq_len=7, reverse=True)

model, log = optimize_model(calcium_data, model, k_splits=7, **kwargs)

plt.figure()

plt.plot(log["train_mask"].to(float).numpy())

plt.title("Train mask")

plt.xlabel("Time")

plt.ylabel("Test (0) / Train (1)")

plt.show()

config = OmegaConf.load("conf/train.yaml")

model, log_dir = train_model(model, dataset, config, shuffle=True)

plot_before_after_weights(log_dir)
