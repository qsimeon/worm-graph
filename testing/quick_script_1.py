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
    # create a model
    model = LinearNN(302, 64).double()
    # make predictions with untrained model
    targets, predictions = model_predict(model, calcium_data)
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")
    # figure of untrained model weights and neuron 0 calcium traces
    fig, axs = plt.subplots(2, 1)
    # model untrained weights
    axs[0].imshow(model.linear.weight.detach().cpu().T)
    axs[0].set_title("Model initial readout weights")
    axs[0].set_xlabel("Output size")
    axs[0].set_ylabel("Input size")
    # neuron 5 targets and predictions
    axs[1].plot(targets[:, 0], label="target")
    axs[1].plot(predictions[:, 0], alpha=0.8, label="prediction")
    axs[1].legend()
    axs[1].set_title("Neuron 0 target and prediction")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("$Ca^{2+} \Delta F / F$")
    plt.show()
