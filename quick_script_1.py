"""
Testing out prediction with untrained 
model on data from a single worm.
"""
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models._utils import NeuralCFC
from train._utils import model_predict
from data._main import get_dataset

config = OmegaConf.load("conf/dataset.yaml")

if __name__=="__main__":
    # load a dataset (containes multiple worms)
    dataset = get_dataset(config)
    # get the calcium data for one worm
    single_worm_dataset = dataset["worm0"]
    calcium_data = single_worm_dataset["calcium_data"]
    # create a model
    model = NeuralCFC(302, 64).double()
    # make predictions with untrained model
    targets, predictions = model_predict(model, calcium_data)
    print("Targets:", targets.shape, "\nPredictions:", predictions.shape, end="\n\n")
    # figure of untrained model weights
    fig, ax = plt.subplots(1, 1)
    ax.imshow(model.linear.weight.detach().cpu().T)
    ax.set_title("Model readout weights")
    ax.set_xlabel("Output size")
    ax.set_ylabel("Input size")
    plt.show()

    
