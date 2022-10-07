#!/usr/bin/env python
# coding: utf-8

# In[2]:


import io
import json
import numpy as np
from six.moves import urllib
from preprocessing import StaticGraphTemporalSignal
import os 
import pandas as pd
os.chdir('/Users/helenahu/Desktop/PyTorch_Implementation/data')

class WormNeuralDynamicsDataLoader(object):
    def get_edges_weights_features_targets(self):
        lags = 2
        E = -48.0 * np.load('emask_default.npy')  # N x 1 directionality vector
        Gg_Static = np.load('Gg_default.npy')  # N X N matrix of gap junction connections
        Gs_Static = np.load('Gs_default.npy')  # N X N matrix of synaptic junction connections

        overlap=0
        just_g=0
        just_s=0
        rc_with_connection = []
        new_edge_weights = []
        for r in range(Gg_Static.shape[0]):
            for c in range(Gg_Static.shape[1]):
                if (Gg_Static[r,c]>0) and (Gs_Static[r,c]>0):
                    overlap+=1
                    if (r,c) not in rc_with_connection:
                        rc_with_connection.append((r,c))
                        new_edge_weights.append(Gs_Static[r,c]+Gg_Static[r,c])
                elif (Gg_Static[r,c]>0) and (Gs_Static[r,c]==0): 
                    just_g+=1
                    if (r,c) not in rc_with_connection:
                        rc_with_connection.append((r,c))
                        new_edge_weights.append(Gg_Static[r,c])
                elif (Gs_Static[r,c]>0) and (Gg_Static[r,c]==0):
                    just_s+=1
                    if (r,c) not in rc_with_connection:
                        rc_with_connection.append((r,c))
                        new_edge_weights.append(Gs_Static[r,c])

        print('Only gap junctions:' ,just_g)
        print('Only synaptic junctions:' ,just_s)         
        print('Both gap and synaptic junctions:' ,overlap)
        self.edges = np.array(rc_with_connection).T
        self.edge_weights = np.array(new_edge_weights)

        saved_dynamics_array = pd.read_csv('saved_dynamics_array.csv', index_col=0)
        stacked_target = np.array(saved_dynamics_array[saved_dynamics_array['run'] == 0].pivot(index='timestep', columns='neuron', values='voltage'))
        self.features = [
            np.concatenate((stacked_target[i : i + lags, :].T, E), axis=1)
            for i in range(stacked_target.shape[0] - lags)]

        self.targets = [
            stacked_target[i + lags, :].T
            for i in range(stacked_target.shape[0] - lags)]
    def get_dataset(self):
        self.get_edges_weights_features_targets()
        dataset = StaticGraphTemporalSignal.StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset

