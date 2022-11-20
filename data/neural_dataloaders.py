import torch
import os
import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from map_dataset import MapDataset
from batch_sampler import BatchSampler
from torch_geometric.data import download_url, extract_zip

def download_neural_datasets():
    url = 'https://www.dropbox.com/s/9dnzrlh12hf5p89/opensource_data.zip?dl=1'
    filename = os.path.join('opensource_data.zip')
    download_url(url=url, folder=os.getcwd(), filename=filename)
    extract_zip(filename, folder=os.getcwd()) # extract zip file
    os.unlink(filename) # remove zip file

def load_Nguyen2017():
    '''
    Args: Loads the worm neural activity datasets from Nguyen et al., PLOS CompBio 2017, 
    Automatically tracking neurons in a moving and deforming brain.
    '''
    # load the raw data
    arr = loadmat('Nguyen2017/heatData_worm2.mat') # load .mat file
    print(list(arr.keys())[3:])
    G2 = arr['G2'] # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr['cgIdx'].squeeze() # ordered indices derived from heirarchically clustering the correlation matrix. 
    # this dataset does not have neuron names
    real_data = G2[cgIdx-1, :].T # to show organized traces, use Ratio2(cgIdx,:)
    real_data = np.nan_to_num(real_data) # replace NaNs 
    worm_IDs = None
    max_time, num_neurons = real_data.shape 
    print('len. Ca recording %s, num. neurons %s'%(max_time, num_neurons))
    # normalize the data 
    sc = preprocessing.MinMaxScaler()
    real_data = sc.fit_transform(real_data[:, :num_neurons]) 
    # return data
    return real_data, worm_IDs

def load_Kaplan2020():
    '''
    Args: Loads the worm neural activity datasets from Kaplan et al., Neuron 2020, 
    Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales.
    '''
    # load .mat file
    arr = mat73.loadmat('Kaplan2020/Neuron2019_Data_RIShisCl.mat')['RIShisCl_Neuron2019']
    print(list(arr.keys()))
    # get data for all worms
    all_IDs = arr['neuron_ID'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces_bleach_corrected'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    # get the neuron IDs and calcium of one worm
    real_data = all_traces[0]
    worm_IDs = [a for a in all_IDs[0] if not a.isnumeric()] 
    # filter for named neurons
    worm_IDs = {id: j for id,j in enumerate(worm_IDs) if isinstance(j,str)} 
    max_time, num_neurons = real_data.shape  
    print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(max_time, num_neurons, len(worm_IDs)))
    # normalize the data 
    sc = preprocessing.MinMaxScaler()
    real_data = sc.fit_transform(real_data[:, :num_neurons]) 
    # return data
    return real_data, worm_IDs

def load_Uzel2022():
    '''
    Args: Loads the worm neural activity datasets from Uzel et al 2022., Cell CurrBio 2022, 
    A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans.
    '''
    # load the raw data
    arr = mat73.loadmat('Uzel2022/Uzel_WT.mat')['Uzel_WT'] # load .mat file
    print(list(arr.keys()))
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    # get the neuron IDs and calcium of one worm
    real_data = all_traces[0]
    worm_IDs = [np.array(a).item() for a in all_IDs[0]] 
    # filter for named neurons
    worm_IDs = {id: j for id, j in enumerate(worm_IDs) if isinstance(j,str)} 
    max_time, num_neurons = real_data.shape  
    print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(max_time, num_neurons, len(worm_IDs)))
    # normalize the data 
    sc = preprocessing.MinMaxScaler()
    real_data = sc.fit_transform(real_data[:, :num_neurons]) 
    # return data
    return real_data, worm_IDs

if __name__=='__main__':
    # load the dataset
    download_neural_datasets()
    DATA, IDs = load_Uzel2022()
    max_time, num_neurons = DATA.shape  
    # plot data
    plt.figure()
    inds = np.random.choice(range(num_neurons), 10)
    plt.plot(DATA[:200, inds])
    plt.legend([IDs[i] for i in inds], title='neuron ID', loc='upper right', fontsize='x-small', title_fontsize='small')
    plt.xlabel('time')
    plt.ylabel('$\delta F / F$')
    plt.title('Calcium traces (first 200 timesteps) of 10 C. elegans neurons')
    plt.show()
    # data loader
    dataset = MapDataset(DATA, neurons=[0,1,2], tau=5, size=200, feature_mask=torch.tensor([1,1] + 8*[0]).to(torch.bool))
    print('size', dataset.size, 'feature', dataset.num_features)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=BatchSampler(dataset.batch_indices)) # shuffle and sampler must be None
    # testing our data-loader
    gen = iter(loader)
    X, Y, meta = next(gen) 
    print(X.shape, Y.shape, {k: meta[k][0] for k in meta}, list(map(lambda x: x.shape, meta.values()))) # each batch contains all samples of a fixed length