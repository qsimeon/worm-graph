import torch
from torch_geometric.data import download_url, extract_zip
import os
import shutil
import mat73
import pickle
import numpy as np
from scipy.io import loadmat
from utils import ROOT_DIR, VALID_DATASETS
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer


def pickle_neural_data(url, zipfile, dataset="all", transform=MinMaxScaler(feature_range=(-1,1))):
    '''
    Function for converting C. elegans neural data from open source 
    datasets into a consistent and compressed form (.pickle files) for
    our purposes. 
    url: str, a download link to a zip file containing the opensource data in raw form.
    zipfile: str, the name of the zipfile that is being downloaded.
    '''
    global source_path, processed_path
    root = ROOT_DIR
    zip_path = os.path.join(root, zipfile)
    source_path = os.path.join(root, zipfile.strip('.zip'))
    processed_path = os.path.join(root, "data/processed/neural")
    transform = PowerTransformer(standardize=False, method="yeo-johnson")
    # Download the curated open-source worm datasets from host server
    if not os.path.exists(source_path):
        # Downloading can take up to 8 minutes depending on your network speed!
        download_url(url=url, folder=root, filename=zipfile)
        extract_zip(zip_path, folder=source_path) # extract zip file
        os.unlink(zip_path) # remove zip file
    # Pickle all the datasets?
    if dataset is None or dataset.lower()=="all":
        for dataset in VALID_DATASETS:
            pickler = eval('pickle_'+dataset)
            pickler(transform)
    # (re)-Pickle a single dataset
    else:
        assert dataset in VALID_DATASETS, "Invalid dataset requested! Please pick one from:\n{}".format(list(VALID_DATASETS))
        pickler = eval('pickle_'+dataset)
        pickler(transform)
    return None

def pickle_Kato2015(transform):
    '''
    Pickles the worm neural activity data from Kato et al., Cell Reports 2015, 
    Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans.
    '''
    data_dict = dict()
    # 'WT_Stim'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Kato2015', 'WT_Stim.mat'))['WT_Stim']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for i, real_data in enumerate(all_traces):
        worm =  "worm"+str(i)
        i_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[i]]
        i_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(i_IDs)]
        _, inds = np.unique(i_IDs, return_index=True)
        i_IDs = [i_IDs[_] for _ in inds]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(i_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items()) # map should be neuron -> index
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = transform() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # 'WT_NoStim'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Kato2015', 'WT_NoStim.mat'))['WT_NoStim']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['NeuronNames'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['deltaFOverF_bc'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for ii, real_data in enumerate(all_traces):
        worm =  "worm"+str(ii + i+1)
        ii_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[ii]]
        ii_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(ii_IDs)]
        _, inds = np.unique(ii_IDs, return_index=True)
        ii_IDs = [ii_IDs[_] for _ in inds]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(ii_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # pickle the data
    file = os.path.join(processed_path, "Kato2015.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kato2015 = pickle.load(pickle_in)
    print()
    print(Kato2015.keys())
    print()

def pickle_Nichols2017():
    '''
    Pickles the worm neural activity data from Nichols et al., Science 2017, 
    A global brain state underlies C. elegans sleep behavior.
    '''
    data_dict = dict()
    # 'n2_let'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Nichols2017', 'n2_let.mat'))['n2_let']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for i, real_data in enumerate(all_traces):
        worm =  "worm"+str(i)
        i_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[i]]
        i_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(i_IDs)]
        _, inds = np.unique(i_IDs, return_index=True)
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(i_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # 'n2_prelet'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Nichols2017', 'n2_prelet.mat'))['n2_prelet']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for ii, real_data in enumerate(all_traces):
        worm =  "worm"+str(ii + i+1)
        ii_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[ii]]
        ii_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(ii_IDs)]
        _, inds = np.unique(ii_IDs, return_index=True)
        ii_IDs = [ii_IDs[_] for _ in inds]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(ii_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # 'npr1_let'
    # load the third .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Nichols2017', 'npr1_let.mat'))['npr1_let']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for iii, real_data in enumerate(all_traces):
        worm =  "worm"+str(iii + ii+1 + i+1)
        iii_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[iii]]
        iii_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(iii_IDs)]
        _, inds = np.unique(iii_IDs, return_index=True)
        iii_IDs = [iii_IDs[_] for _ in inds]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(iii_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # 'npr1_prelet'
    # load the fourth .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Nichols2017', 'npr1_prelet.mat'))['npr1_prelet']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for iv, real_data in enumerate(all_traces):
        worm =  "worm"+str(iv + iii+1 + ii+1 + i+1)
        iv_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[iv]]
        iv_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(iv_IDs)]
        _, inds = np.unique(iv_IDs, return_index=True)
        iv_IDs = [iv_IDs[_] for _ in inds]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(iv_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # pickle the data
    file = os.path.join(processed_path, "Nichols2017.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nichols2017 = pickle.load(pickle_in)
    print()
    print(Nichols2017.keys())
    print()

def pickle_Nguyen2017():
    '''
    Pickles the worm neural activity data from Nguyen et al., PLOS CompBio 2017, 
    Automatically tracking neurons in a moving and deforming brain.
    '''
    # WORM 0
    # load .mat file for  worm 0
    arr0 = loadmat(os.path.join(source_path, 'Nguyen2017', 'heatData_worm0.mat')) # load .mat file
    print(list(arr0.keys()))
    print()
    # get data for worm 1
    G2 = arr0['G2'] # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr0['cgIdx'].squeeze() # ordered indices derived from heirarchically clustering the correlation matrix. 
    real_data0 = G2[cgIdx-1, :].T # to show organized traces, use Ratio2(cgIdx,:)
    real_data0 = np.nan_to_num(real_data0) # replace NaNs 
    max_time0, num_neurons0 = real_data0.shape 
    num_named0 = 0
    worm0_ID = {i: str(i) for i in range(num_neurons0)}
    worm0_ID = dict((v,k) for k,v in worm0_ID.items())
    print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time0, num_neurons0, num_named0))
    print()
    # normalize the data 
    sc = MinMaxScaler()
    real_data0 = sc.fit_transform(real_data0[:, :num_neurons0]) 
    # add a feature dimension and convert to tensor
    real_data0 = np.expand_dims(real_data0, axis=-1)
    real_data0 = torch.tensor(real_data0, dtype=torch.float64)
    # WORM 1
    # load .mat file for  worm 1
    arr1 = loadmat(os.path.join(source_path, 'Nguyen2017', 'heatData_worm1.mat')) # load .mat file
    print(list(arr1.keys()))
    print()
    # get data for worm 1
    G2 = arr1['G2'] # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr1['cgIdx'].squeeze() # ordered indices derived from heirarchically clustering the correlation matrix. 
    real_data1 = G2[cgIdx-1, :].T # to show organized traces, use Ratio2(cgIdx,:)
    real_data1 = np.nan_to_num(real_data1) # replace NaNs 
    max_time1, num_neurons1 = real_data1.shape 
    num_named1 = 0
    worm1_ID = {i: str(i) for i in range(num_neurons1)}
    worm1_ID = dict((v,k) for k,v in worm1_ID.items())
    print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time1, num_neurons1, num_named1))
    print()
    # normalize the data 
    sc = MinMaxScaler()
    real_data1 = sc.fit_transform(real_data1[:, :num_neurons1]) 
    # add a feature dimension and convert to tensor
    real_data1 = np.expand_dims(real_data1, axis=-1)
    real_data1 = torch.tensor(real_data1, dtype=torch.float64)
    # WORM 2
    # load .mat file for  worm 1
    arr2 = loadmat(os.path.join(source_path, 'Nguyen2017', 'heatData_worm2.mat')) # load .mat file
    print(list(arr2.keys()))
    print()
    # get data for worm 2
    G2 = arr2['G2'] # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr2['cgIdx'].squeeze() # ordered indices derived from heirarchically clustering the correlation matrix. 
    real_data2 = G2[cgIdx-1, :].T # to show organized traces, use Ratio2(cgIdx,:)
    real_data2 = np.nan_to_num(real_data2) # replace NaNs 
    max_time2, num_neurons2 = real_data2.shape 
    num_named2 = 0
    worm2_ID = {i: str(i) for i in range(num_neurons2)}
    worm2_ID = dict((v,k) for k,v in worm2_ID.items())
    print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time2, num_neurons2, num_named2))
    print()
    # normalize the data 
    sc = MinMaxScaler()
    real_data2 = sc.fit_transform(real_data2[:, :num_neurons2]) 
    # add a feature dimension and convert to tensor
    real_data2 = np.expand_dims(real_data2, axis=-1)
    real_data2 = torch.tensor(real_data2, dtype=torch.float64)
    # pickle the data
    data_dict = {
            'worm0': {'data': real_data0, 'neuron_id': worm0_ID, 'max_time': max_time0, 
                        'num_neurons': num_neurons0, 'num_named': num_named0}, 
            'worm1': {'data': real_data1, 'neuron_id': worm1_ID, 'max_time': max_time1, 
                        'num_neurons': num_neurons1, 'num_named': num_named1}, 
            'worm2': {'data': real_data2, 'neuron_id': worm2_ID, 'max_time': max_time2, 
                        'num_neurons': num_neurons2, 'num_named': num_named2},
                }
    file = os.path.join(processed_path, "Nguyen2017.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nguyen2017 = pickle.load(pickle_in)
    print(Nguyen2017.keys())
    print()

def pickle_Skora2018():
    '''
    Pickles the worm neural activity data from Skora et al., Cell Reports 2018, 
    Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans. 
    '''
    data_dict = dict()
    # 'WT_fasted'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Skora2018', 'WT_fasted.mat'))['WT_fasted']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for i, real_data in enumerate(all_traces):
        worm =  "worm"+str(i)
        i_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[i]]
        i_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(i_IDs)]
        _, inds = np.unique(i_IDs, return_index=True)
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(i_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # 'WT_starved'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Skora2018', 'WT_starved.mat'))['WT_starved']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for ii, real_data in enumerate(all_traces):
        worm =  "worm"+str(ii + i+1)
        ii_IDs = [(j[0] if isinstance(j,list) else j) for j in all_IDs[ii]]
        ii_IDs = [(str(_) if j is None or isinstance(j, np.ndarray) else str(j)) for _,j in enumerate(ii_IDs)]
        _, inds = np.unique(ii_IDs, return_index=True)
        ii_IDs = [ii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j)) for nid,j  in enumerate(ii_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # pickle the data
    file = os.path.join(processed_path, "Skora2018.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Skora2018 = pickle.load(pickle_in)
    print()
    print(Skora2018.keys())
    print()

def pickle_Kaplan2020():
    '''
    Pickles the worm neural activity data from Kaplan et al., Neuron 2020, 
    Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales.
    '''
    data_dict = dict()
    # 'RIShisCl_Neuron2019'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Kaplan2020', 'Neuron2019_Data_RIShisCl.mat'))['RIShisCl_Neuron2019']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['neuron_ID'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces_bleach_corrected'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for i, real_data in enumerate(all_traces):
        worm =  "worm"+str(i)
        _, inds = np.unique(all_IDs[i], return_index=True) # only keep indices of unique neuron IDs
        all_IDs[i] = [all_IDs[i][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: str(j) for nid, j in enumerate(all_IDs[i])} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 
                                'num_named': num_named},
                        })
    # 'MNhisCl_RIShisCl_Neuron2019'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Kaplan2020', 'Neuron2019_Data_MNhisCl_RIShisCl.mat'))['MNhisCl_RIShisCl_Neuron2019']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['neuron_ID'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces_bleach_corrected'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for ii, real_data in enumerate(all_traces):
        worm =  "worm"+str(ii + i+1)
        _, inds = np.unique(all_IDs[ii], return_index=True) # only keep indices of unique neuron IDs
        all_IDs[ii] = [all_IDs[ii][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: str(j) for nid, j in enumerate(all_IDs[ii])} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 
                                'num_named': num_named},
                        })
    # 'MNhisCl_RIShisCl_Neuron2019'
    # load the third .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Kaplan2020', 'Neuron2019_Data_SMDhisCl_RIShisCl.mat'))['SMDhisCl_RIShisCl_Neuron2019']
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['neuron_ID'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces_bleach_corrected'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for iii, real_data in enumerate(all_traces):
        worm =  "worm"+str(iii + ii+1 + i+1)
        _, inds = np.unique(all_IDs[iii], return_index=True) # only keep indices of unique neuron IDs
        all_IDs[iii] = [all_IDs[iii][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: str(j) for nid, j in enumerate(all_IDs[iii])} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 
                                'num_named': num_named},
                        })
    # pickle the data
    file = os.path.join(processed_path, "Kaplan2020.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kaplan2020 = pickle.load(pickle_in)
    print()
    print(Kaplan2020.keys())
    print()

def pickle_Uzel2022():
    '''
    Pickles the worm neural activity data from Uzel et al 2022., Cell CurrBio 2022, 
    A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans.
    '''
    data_dict = dict()
    # load .mat file
    arr = mat73.loadmat(os.path.join(source_path, 'Uzel2022', 'Uzel_WT.mat'))['Uzel_WT'] # load .mat file
    print(list(arr.keys()))
    print()
    # get data for all worms
    all_IDs = arr['IDs'] # identified neuron IDs (only subset have neuron names)
    all_traces = arr['traces'] # neural activity traces corrected for bleaching
    print('num. worms:', len(all_IDs))
    print()
    for i, real_data in enumerate(all_traces):
        worm =  "worm"+str(i)
        i_IDs = [np.array(j).item() for j in all_IDs[i]]
        _, inds = np.unique(i_IDs, return_index=True) # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]
        neuron_ID = {nid: (str(int(j)) if type(j)!=str else j) for nid,j  in enumerate(i_IDs)} 
        neuron_ID = {nid: (name.replace('0','') if not name.endswith('0') and not name.isnumeric() else name) for nid, name in neuron_ID.items()}
        neuron_ID = dict((v,k) for k,v in neuron_ID.items())
        max_time, num_neurons = real_data.shape  
        num_named = len([k for k in neuron_ID.keys() if not k.isnumeric()]) # number of neurons that were ID'd
        print("len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"%(
            max_time, num_neurons, num_named))
        sc = MinMaxScaler() # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons]) 
        real_data = np.expand_dims(real_data, axis=-1)
        real_data = torch.tensor(real_data, dtype=torch.float64) # add a feature dimension and convert to tensor
        data_dict.update({worm: {'data': real_data, 'neuron_id': neuron_ID, 
                                'max_time': max_time, 'num_neurons': num_neurons, 'num_named': num_named},
                        })
    # pickle the data
    file = os.path.join(processed_path, "Uzel2022.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Uzel2022 = pickle.load(pickle_in)
    print(Uzel2022.keys())
    print()

if __name__=='__main__':
    url = 'https://www.dropbox.com/s/l3pedwweqqsmd38/opensource_data.zip?dl=1'
    zipfile = 'opensource_data.zip'
    # pickle a particular dataset
    dataset = 'Skora2018'
    pickle_neural_data(url=url, zipfile=zipfile, dataset=dataset)
    # pickle all the datasets
    pickle_neural_data(url=url, zipfile=zipfile, dataset="all")
    # delete the downloaded raw datasets.
    shutil.rmtree(source_path) # files too large to push to GitHub