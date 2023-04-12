#!/usr/bin/env python
# encoding: utf-8
'''
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _trial.py
@time: 2023/4/11 19:05
'''

from analysis._utils import *
from utils import *

# load your trained model checkpoint of choice
PATH = "logs/hydra/199_epochs_19303_worms.pt"
checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))
for key in checkpoint.keys():
    print(key)

# get checkpoint variables
model_name = checkpoint["model_name"]
dataset_name = checkpoint["dataset_name"]
input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
model_state_dict = checkpoint["model_state_dict"]
optimizer_state_dict = checkpoint["optimizer_state_dict"]
num_layers = checkpoint["num_layers"]
smooth_data = checkpoint["smooth_data"]
epoch = checkpoint["epoch"]

print("{} model was trained on dataset {} for {} epochs.".format(model_name, dataset_name, epoch))

# load model checkpoint
model = eval(model_name)(input_size, hidden_size, num_layers)
model.load_state_dict(model_state_dict)
model.eval()
print(model)

config = OmegaConf.load("conf/dataset.yaml")

dataset = get_dataset(config)

# neurons_on_category(model, dataset, True, 1)
# print("end")

plot_trailing_loss_vs_parameter_legend("logs/single_dataset", "train.train_size", "dataset.name")
print("double end")
