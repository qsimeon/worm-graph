from data._main import get_dataset

from omegaconf import OmegaConf

from train._utils import optimize_model

from models._utils import LinearNN

import matplotlib.pyplot as plt

dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))

calcium_data = dataset["worm0"]["calcium_data"]

model = LinearNN(302, 64).double()

model, log = optimize_model(calcium_data, model, k_splits=7)

plt.figure()

plt.plot(log["train_mask"].to(float).numpy())

plt.show()
