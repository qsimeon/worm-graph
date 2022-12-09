"""
This script runs a prediction task for for all the different type of GNN models inside
"gnn_models.py", for every worm in all the different datasets in "load_neural_activity.py"
"""

from data.load_connectome import CElegansDataset
from tasks.all_tasks import OneStepPrediction
from train.train_gnn import optimize_model
from train.train_gnn import model_predict
import matplotlib.pyplot as plt
import pandas as pd
import csv

#loading all models
from models.gnn_models import *
#loading all datasets
from data.load_neural_activity import *

def runModel(modelName, wormSet, worm):
    # load conectome
    graph = CElegansDataset()[0]
    # load data
    set = wormSet(); single_worm_dataset = set[worm]
    dataset, neuron_ids = single_worm_dataset['data'], single_worm_dataset['neuron_ids']
    # define the task
    task = OneStepPrediction(graph, dataset, neuron_ids)
    # construct the model
    if modelName == EvolveRCGN:
        model = modelName(node_count=graph.num_nodes, node_features=task.seq_len)
    elif modelName == GatedRGCN:
        model = modelName(node_features=task.seq_len)
    elif modelName == DeepRGCN:
        model = modelName(node_features=task.seq_len)
    
    # train the model
    model, log = optimize_model(task, model)
    preds = model_predict(task, model)
    return model, log, preds

if __name__ == "__main__":
    #making a list of all available models, the classes in "gnn_models.py"
    models = [EvolveRCGN, GatedRGCN, DeepRGCN]
    #making a list of all worm datasets
    wormSets = [load_Uzel2022, load_Kaplan2020, load_Nguyen2017]
    worms = ['worm1', 'worm2']

    for modelName in models:
        for wormSet in wormSets:
            for worm in worms:
                model, log, preds = runModel(modelName,wormSet,worm)

                #save a csv files named for the worm and modelName, containing model, log, and preds
                #model.to_csv(worm+modelName.__name__+'model.csv')
                x_df = pd.DataFrame(preds)
                x_df.to_csv(logs + '/' + wormSet.__name__+worm+modelName.__name__+'preds.csv')
                x_df = pd.DataFrame(log)
                x_df.to_csv(logs + '/' + wormSet.__name__+worm+modelName.__name__+'log.csv')