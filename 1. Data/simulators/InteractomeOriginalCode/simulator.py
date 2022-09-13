import numpy as np
import pandas as pd
import random
from os.path import join as oj
import neural_interactome
import networkx as nx

def get_conn_ids(adj_mat):
    '''returns source and target ids'''
    graph = nx.from_numpy_array(adj_mat)
    graph_edges = np.array(graph.edges())
    source_id,target_id = graph_edges[:,0],graph_edges[:,1]
    return(source_id,target_id, graph)
    
    
def GenerateRecordings(NumTimesteps, PercentPerturbed, AmountPerturbed, sim_dir):

    E = -48.0 * np.load(oj(sim_dir, 'emask_default.npy'))  # N x 1 directionality vector
    Gg_Static = np.load(oj(sim_dir, 'Gg_default.npy'))  # N X N matrix of gap junction connections
    Gs_Static = np.load(oj(sim_dir, 'Gs_default.npy'))  # N X N matrix of synaptic junction connections
    txt_file = oj(sim_dir, 'neuron_names_cleaned.txt')
    with open(txt_file) as f:
        neuronNames = f.read().splitlines() 
    
    """
    Parameters
    ----------
    - NumTimesteps : Int, number of timesteps in each recording
    - PercentPerturbed : 0-1 float, Percent of total neurons given inital activation
    - AmountPerturbed: 0-1 float, Initial activation for perturbed neurons
    - sim_dir : simulator location
    """
    NumNeurons = 279
    NumPerturbed = int(NumNeurons // (1/PercentPerturbed))
    
    saved_dynamics_array=pd.DataFrame()
    df=pd.DataFrame()
    
    #write statics
    source_id_electric,target_id_electric,saved_elec_graph = get_conn_ids(Gg_Static)
    source_id_chemical,target_id_chemical,saved_chem_graph = get_conn_ids(Gs_Static)

    class statics:
        elec_graph = saved_elec_graph
        chem_graph = saved_chem_graph
        directionality_E = E
        names = neuronNames

    #some function that runs existing simulation on github with randomized starting parameters
    #and outputs saved_dynamics
    #generate random inital conditions
    input_Array = np.zeros(Gg_Static.shape[0])
    chosenNeurons = random.sample(range(NumNeurons), NumPerturbed)
    print("Chosen Neurons"+str(chosenNeurons))
    input_Array[chosenNeurons] = AmountPerturbed
    
    #run sim
    outputs = neural_interactome.run_NI_sim(input_Array, Gg_Static, Gs_Static, E, max_time = NumTimesteps)   
    #print(out[200:,:].shape)
    #np.savetxt('data/' +'run1cutoff.csv', out[200:,:], delimiter=",")
    #np.savetxt('data/' +'run1cutoff.csv', out, delimiter=",")

    class out:
        voltages = outputs
        targets2 = None
    return out, statics