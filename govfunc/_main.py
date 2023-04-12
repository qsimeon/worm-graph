#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _main.py
@time: 2023/2/28 12:15
"""

from govfunc._utils import *


def main(
    dataset: dict,
    config: DictConfig,
    folder,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    shuffle: bool = False,
):
    # single_worm_dataset = dataset["worm0"]
    # calcium_data = dataset["worm0"]["calcium_data"]
    # residual = dataset["worm0"]["residual_calcium"]
    #
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
    for i in range(0, len(dataset)):
        worm = "worm" + str(i)
        name_mask = dataset[worm]["named_neurons_mask"]
        calcium_data = dataset[worm]["calcium_data"][:]
        residual = dataset[worm]["residual_calcium"][:]
        start = config.govfunc.start
        tau = config.govfunc.tau
        seq_len = dataset[worm]["max_time"] - tau - start - 1
        assert (
            seq_len + start + tau < dataset[worm]["max_time"]
        ), "exceed the max_time length"

        # cal to res, tau = 1
        x = calcium_data[start : start + seq_len]
        residual = residual[start + tau : start + seq_len + tau]
        dx_nan = torch.div(
            residual, dataset[worm]["dt"][start + tau : start + seq_len + tau]
        )
        dx = torch.where(torch.isnan(dx_nan), torch.full_like(dx_nan, 0), dx_nan)

        # # cal to cal, tau = 10, start:end = start:(start+seq_len)
        # x = calcium_data[start:start + seq_len]
        # dx = calcium_data[start + tau: start + tau + seq_len]

        # the original status x0 from time step 0
        x0 = x[0]
        Theta = generate_polynomial(x, polyorder=1, usesine=False)
        # TODO: figure out the influence of changing lambda_
        lambda_ = 0.025
        Xi = sparsifyDynamics(Theta, dx, lambda_, x.shape[1])
        print("Xi: ", Xi.shape)

        slot_x = []
        slot_y = ["offset"]
        for i in range(0, name_mask.shape[0]):
            slot_x.append(dataset[worm]["slot_to_named_neuron"][i])
            slot_y.append(dataset[worm]["slot_to_named_neuron"][i])

        slot_x = np.array(slot_x)
        slot_y = np.array(slot_y)

        # index = list(range(0, 302))

        I = pd.Index(slot_y, name="rows")
        C = pd.Index(slot_x, name="cols")
        # I = pd.Index(index, name="rows")
        # index_y = list(range(-1, 302, 1))
        # C = pd.Index(index_y, name="cols")
        data = pd.DataFrame(Xi, index=I, columns=C)

        parent_path = os.getcwd() + "/govfunc" + folder
        isExist = os.path.exists(parent_path)
        if not isExist:
            os.mkdir(parent_path)

        path = (
            os.getcwd() + "/govfunc" + folder + "/coefficient_CalToRes_tau_" + str(tau)
        )
        isExist = os.path.exists(path)
        if not isExist:
            os.mkdir(path)
        data.to_hdf(
            "./govfunc"
            + folder
            + "/coefficient_CalToRes_tau_"
            + str(tau)
            + "/coef_"
            + worm
            + ".hdf",
            "test",
        )
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
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    f = "/" + dataset["worm0"]["dataset"]
    main(dataset, config, folder=f)

    for i in range(0, len(dataset)):
        data, sorted_np = coef_analysis(
            dataset_name=dataset["worm0"]["dataset"],
            worm_name="worm" + str(i),
            n_cluster=3,
            folder=f + "/coefficient_CalToRes_tau_" + str(config.govfunc.tau),
        )
        print(sorted_np)
