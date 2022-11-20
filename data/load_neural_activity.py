import os
import pickle
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
    pass
    # DATA, IDs = load_Uzel2022()
    # max_time, num_neurons = DATA.shape  
    # # data loader
    # dataset = MapDataset(DATA, neurons=[0,1,2], tau=5, size=200, feature_mask=torch.tensor([1,1] + 8*[0]).to(torch.bool))
    # print('size', dataset.size, 'feature', dataset.num_features)
    # loader = torch.utils.data.DataLoader(dataset, batch_sampler=BatchSampler(dataset.batch_indices)) # shuffle and sampler must be None
    # # testing our data-loader
    # gen = iter(loader)
    # X, Y, meta = next(gen) 
    # print(X.shape, Y.shape, {k: meta[k][0] for k in meta}, list(map(lambda x: x.shape, meta.values()))) # each batch contains all samples of a fixed length