from models._utils import NeuralCFC

from train._utils import model_predict

from omegaconf import OmegaConf

from data._main import get_dataset

import matplotlib.pyplot as plt

config = OmegaConf.load("conf/dataset.yaml")

dataset = get_dataset(config)

single_worm_dataset = dataset["worm0"]

calcium_data = single_worm_dataset["calcium_data"]

model = NeuralCFC(302, 64).double()

fig, ax = plt.subplots(1, 1)

ax.imshow(model.linear.weight.detach().cpu().T)

ax.set_title("Model readout weights")

ax.set_xlabel("Output size")

ax.set_ylabel("Input size")

plt.show()

targets, predictions = model_predict(model, calcium_data)

print("Targets:", targets.shape, "\nPredictions:", predictions.shape)
