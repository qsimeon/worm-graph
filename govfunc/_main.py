#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _main.py
@time: 2023/2/28 12:15
"""
import pandas as pd

from train._main import *
import os


def main(model: torch.nn.Module,
         dataset: dict,
         config: DictConfig,
         optimizer: Union[torch.optim.Optimizer, None] = None,
         shuffle: bool = False,
         ):
    # single_worm_dataset = dataset["worm0"]
    # calcium_data = dataset["worm0"]["calcium_data"]
    # residual = dataset["worm0"]["residual_calcium"]
    #
    # seq_len = config.govfunc.seq_len
    # tau = config.govfunc.tau
    #
    # if config.govfunc.smooth_data:
    #     key_data = "smooth_calcium_data"
    # else:
    #     key_data = "calcium_data"
    #
    # x = []
    # dx = []
    # for i in range(0, single_worm_dataset["max_time"] - seq_len):
    #     x.append(calcium_data[i:i + seq_len])
    #     dx_nan = torch.div(residual[i:i + seq_len], single_worm_dataset["dt"][i:i + seq_len])
    #     dx_notnan = torch.where(torch.isnan(dx_nan), torch.full_like(dx_nan, 0), dx_nan)
    #     dx.append(dx_notnan)
    #
    # x = torch.stack(x)
    # dx = torch.stack(dx)
    #
    # # print(x.shape, dx.shape)
    #
    # slice = 20
    # x0 = x[:slice, :, 0]
    # dx0 = dx[:slice, :, 0]
    # x0 = x0.T
    # dx0 = dx0.T
    #
    # print(x0.shape, dx0.shape)
    #
    # lambda_ = 0.25
    # Xi = sparsifyDynamics(x0, dx0, lambda_, x0.shape[1])
    # print("Xi: ", Xi.shape)
    #
    # sns.heatmap(data=Xi, square=True, cmap="RdBu_r", center=0)
    # plt.show()
    for i in range(0, 6):
        worm = "worm" + str(i)

        calcium_data = dataset[worm]["calcium_data"]
        residual = dataset[worm]["residual_calcium"]
        name_mask = dataset[worm]["named_neurons_mask"]
        x = calcium_data[:, name_mask]
        dx = residual[:, name_mask]

        # print("x_initial: ", x.shape)
        # print("dx: ", dx.shape)

        # the original status x0 from time step 0
        x0 = x[0]
        # print("x0: ", x0.shape)

        Theta = generate_polynomial(x, polyorder=1, usesine=False)

        # TODO: figure out the influence of changing lambda_
        lambda_ = 0.025
        Xi = sparsifyDynamics(Theta, dx, lambda_, x.shape[1])
        print("Xi: ", Xi.shape)

        print(dataset[worm].keys())

        slot_x = []
        slot_y = ["offset"]

        for i in range(0, name_mask.shape[0]):
            if name_mask[i] == True:
                slot_x.append(dataset[worm]["slot_to_named_neuron"][i])
                slot_y.append(dataset[worm]["slot_to_named_neuron"][i])

        slot_x = np.array(slot_x)
        slot_y = np.array(slot_y)
        Xi_plot = np.array(Xi)
        print(slot_x.shape, slot_y.shape, Xi.shape)

        I = pd.Index(slot_y, name="rows")

        C = pd.Index(slot_x, name="cols")

        # dict_plot = {}
        # for i in range(0, slot_y.shape[0]):
        #     dict_plot[slot_y[i]] = Xi_plot[i, :]

        data = pd.DataFrame(Xi, index=I, columns=C)



        data.to_hdf("./govfunc/coefficient/coef_" + worm + '.hdf', "test")
    return

    # # true calcium_data: worm[0]["calcium_data"]
    # # denoted as target
    # target = x
    # target = target.T
    # tr, tc = target.size()
    # print("the shape of target: (" + str(tr) + ", " + str(tc) + ")")
    #
    # # model's results - generated from governing functions
    # # denoted as pred
    # pred = governingFuncPredict(x0, Theta, Xi)
    # pred = torch.tensor(pred)
    # pred = pred.T
    # tr, tc = pred.size()
    # print("the shape of pred: (" + str(tr) + ", " + str(tc) + ")")
    #
    # time_slice = 1000
    # neuro_plot(target[:, 0:time_slice], True)
    # neuro_plot(pred[:, 0:time_slice], False)
    #
    # return 0


if __name__ == "__main__":
    config = OmegaConf.load("conf/govfunc.yaml")
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    print("config:", OmegaConf.to_yaml(config), end="\n\n")

    main(model, dataset, config)
