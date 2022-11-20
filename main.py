from data.load_connectome import CElegansDataset
from tasks.all_tasks import OneStepPrediction
from models.gnn_models import EvolveRCGN
from train.train import optimize_model, model_predict
from scipy.io import loadmat
from sklearn import preprocessing
from utils import ROOT_DIR
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load some real data, TODO: put loading of real data in a separate script/module
    arr = loadmat(os.path.join(ROOT_DIR, 'data', 'raw', 'heatData_worm1.mat'))
    Ratio2 = arr['Ratio2'] # the ratio signal is defined as normalized gPhotoCorr/rPhotoCorr
    cgIdx = arr['cgIdx'].squeeze()  # ordered indices from hierarchically clustering correlation matrix
    dataset = np.nan_to_num(Ratio2[cgIdx-1, :]) # shape (num_neurons, num_timesteps)

    # build and train model
    graph = CElegansDataset()[0]
    task = OneStepPrediction(graph, dataset)
    model = EvolveRCGN(node_count=task.node_count, node_features=task.node_features)
    model, log = optimize_model(task, model)
    preds = model_predict(task, model)

    # TODO: put plotting code like this in a separte module file
    plt.figure()
    plt.plot(log['epochs'], log['train_losses'], label='train')
    plt.plot(log['epochs'], log['test_losses'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('EvolveRCGN loss curves')
    plt.legend()
    plt.show()

    plt.figure()
    nid = np.random.choice(task.node_count)
    sc = preprocessing.MinMaxScaler()
    data = sc.fit_transform(task.graph.x.clone().detach().numpy())
    plt.plot(data[nid, :1000], alpha=0.3, label='real')
    plt.gca().set_prop_cycle(None)
    plt.plot(preds[nid, :1000], '--', label='predicted', linewidth=3) 
    plt.xlabel('Time')
    plt.ylabel('Normalized Voltage')
    plt.title('EvolveRCGN prediction for neuron %s'%nid)
    plt.legend()
    plt.show()