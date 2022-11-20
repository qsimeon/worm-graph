import torch
from map_dataset import MapDataset
from batch_sampler import BatchSampler
import os
import pickle
import numpy as np
from utils import ROOT_DIR

def load_Nguyen2017():
    '''
    Loads the worm neural activity datasets from Nguyen et al., PLOS CompBio 2017, 
    Automatically tracking neurons in a moving and deforming brain.
    '''
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, 'preprocessing', 'Nguyen2017.pickle')
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Nguyen2017 = pickle.load(pickle_in)
    return Nguyen2017

def load_Kaplan2020():
    '''
    Args: Loads the worm neural activity datasets from Kaplan et al., Neuron 2020, 
    Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales.
    '''
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, 'preprocessing', 'Kaplan2020.pickle')
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Kaplan2020 = pickle.load(pickle_in)
    return Kaplan2020

def load_Uzel2022():
    '''
    Loads the worm neural activity datasets from Uzel et al 2022., Cell CurrBio 2022, 
    A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans.
    '''
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, 'preprocessing', 'Uzel2022.pickle')
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Uzel2022 = pickle.load(pickle_in)
    return Uzel2022
    
if __name__=='__main__':
    # load a recent dataset
    Uzel2022 = load_Uzel2022()
    # get data for one worm
    single_worm_dataset = Uzel2022['worm1']
    num_neurons = single_worm_dataset['num_neurons']
    neuron_ids = single_worm_dataset['neuron_ids']
    calcium_data = single_worm_dataset['data']
    data = torch.nn.functional.pad(calcium_data, (0,9), 'constant', 0)  # pad feature dimension to 10D
    # create a dataset and data-loader
    feature_mask = torch.tensor([1,1] + 8*[0]).to(torch.bool) # selects 2 features out of 10
    dataset = MapDataset(data, feature_mask=feature_mask)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=BatchSampler(dataset.batch_indices))
    X, Y, meta = next(iter(loader)) 
    # output properties of the dataset and data loader
    print()
    print('size', dataset.size, 'feature', dataset.num_features)
    print()
    print(X.shape, Y.shape, {k: meta[k][0] for k in meta}, list(map(lambda x: x.shape, meta.values())))
    print()