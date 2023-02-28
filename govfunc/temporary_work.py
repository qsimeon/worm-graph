# a copy from _main.py that worked

from data._main import *
from govfunc._utils import *

from scipy.integrate import odeint
import matplotlib.pyplot as plt


def main():
    config = OmegaConf.load("./dataset.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)
    print("----dataset prepared------")

    # take three worms as an example
    numOfWorm = 3
    worm = []
    for i in range(0, numOfWorm):
        worm.append(dataset["worm" + str(i)])
        worm[i]["calcium_data"] = worm[i]["calcium_data"].T
        # print(worm[i]["calcium_data"].shape)
    rows, cols = worm[0]["calcium_data"].size()
    print("the time steps is " + str(rows))  # 302

    dys = []
    for i in range(0, numOfWorm):
        dy = derivative(worm[i]["calcium_data"], 0)
        # print(dy.shape)
        dys.append(dy)

    # here we just use one worm

    # the original status x0
    x0 = worm[0]["calcium_data"][0]
    print("xxx", x0.shape)

    polyorder = 1
    usesine = False
    r, c = dys[0].shape
    Theta = poolData(worm[0]["calcium_data"], c, polyorder, usesine)
    m = Theta.shape[1]

    lambda_ = 0.025
    Xi = sparsifyDynamics(Theta, dys[0], lambda_, c)

    # true calcium_data: worm[0]["calcium_data"]
    # denoted as target
    target = worm[0]["calcium_data"]
    target = target.T
    tr, tc = target.size()
    print("the shape of target: (" + str(tr) + ", " + str(tc) + ")")

    # model's results - generated from governing functions
    # denoted as pred

    pred = governingFuncPredict(x0, Theta, Xi)
    pred = pred.T

    ## TODO: generate pic for true data and the results for Theta*Xi
    neuro_plot(target[0:20, :], True)
    neuro_plot(pred[0:20, :], False)


    return 0


if __name__ == "__main__":
    main()
