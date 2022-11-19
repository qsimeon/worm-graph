import torch
import numpy as np
from utils import DEVICE as device 
from map_dataset import MapDataset
from batch_sampler import BatchSampler


if __name__=='__main__':
    dataset = MapDataset(DATA, feature_mask=torch.tensor([1,1] + 8*[0]).to(torch.bool), size=200)
    dataset = MapDataset(DATA, neurons=[0,1,2], size=200)
    dataset = MapDataset(DATA, tau=5, size=200)
    dataset = MapDataset(DATA, size=200)
    print('size', dataset.size, 'feature', dataset.num_features)
    data_sampler = BatchSampler(dataset.batch_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=data_sampler) # shuffle and sampler must be None
    # testing our data-loader
    gen = iter(loader)
    X, Y, meta = next(gen) 
    print(X.shape, Y.shape, {k: meta[k][0] for k in meta}, 
        list(map(lambda x: x.shape, meta.values()))) # each batch contains all samples of a fixed length
    X, Y, meta = next(gen)
    print(X.shape, Y.shape, {k: meta[k][0] for k in meta}, 
        list(map(lambda x: x.shape, meta.values()))) # we can get variable length sequences!
    print(X.is_cuda, Y.is_cuda) # check if data on GPU