"""
Tests prediction with untrained 
model on data from a single worm.
"""
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import LinearNN
from train._utils import model_predict
from data._main import get_dataset

config = OmegaConf.load("conf/dataset.yaml")

if __name__ == "__main__":
    # load a dataset (containes multiple worms)
    dataset = get_dataset(config)
    # get the calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]
    time_in_seconds = single_worm_dataset.get("time_in_seconds", None)
    # create a model
    model = LinearNN(302, 64).double()
    # make 1-step ahead prediction with untrained model
    pred_tau = 300
    targets, predictions = model_predict(model, calcium_data, tau=pred_tau)
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")
    # pick a neuron idx
    neuron = 0
    # figure of untrained model weights and neuron xx calcium traces
    fig, axs = plt.subplots(2, 1)
    # model untrained weights
    axs[0].imshow(model.linear.weight.detach().cpu().T)
    axs[0].set_title("Model initial readout weights")
    axs[0].set_xlabel("Output size")
    axs[0].set_ylabel("Input size")
    # neuron xx targets and predictions
    axs[1].plot(time_in_seconds, targets[:, neuron], label="target")
    axs[1].plot(time_in_seconds, predictions[:, neuron], alpha=0.8, label="prediction")
    axs[1].legend()
    axs[1].set_title(
        "Neuron %s target and prediction ($\\tau = %s$)" % (neuron, pred_tau)
    )
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("$Ca^{2+} \Delta F / F$")
    plt.show()
