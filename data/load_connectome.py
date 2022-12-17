import os
import torch
from utils import ROOT_DIR
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import download_url, extract_zip
from preprocessing.process_raw import preprocess


class CElegansDataset(InMemoryDataset):
    def __init__(self, root=os.path.join(ROOT_DIR, 'data'), transform=None, pre_transform=None):
        '''Defines CElegansDataset as a subclass of a PyG InMemoryDataset.'''
        super(CElegansDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])   
        
    @property
    def raw_file_names(self):
        '''List of the raw files needed to proceed.'''
        return ['GHermChem_Edges.csv', 'GHermChem_Nodes.csv', 
                'GHermElec_Sym_Edges.csv', 'GHermElec_Sym_Nodes.csv',
                'LowResAtlasWithHighResHeadsAndTails.csv', 'neuron_names.txt'] 
               
    @property
    def processed_file_names(self):
        '''List of the processed files needed to proceed.'''
        return ['data.pt']

    def download(self):
        '''Download the raw zip file if not already retrieved.'''
        # dataset adapted from from Cook et al. (2019) SI5
        url = 'https://www.dropbox.com/s/utwj011wrik7l1j/raw_data.zip?dl=1' # base url
        filename = os.path.join('raw_data.zip')
        folder = os.path.join(self.raw_dir)
        download_url(url=url, folder=os.getcwd(), filename=filename) # download zip file 
        extract_zip(filename, folder=folder) # unzip data into raw directory
        os.unlink(filename) # remove zip file

    def process(self):
        '''Process the raw files and return the dataset (i.e. graph).'''
        # fast preprocess here
        data_path = os.path.join(ROOT_DIR, 'preprocessing', 'graph_tensors.pt')
        if not os.path.exists(data_path):
            print("Building from raw...")
            preprocess(raw_dir=self.raw_dir, raw_files=self.raw_file_names)
        # load the raw data
        print("Loading from preprocess...")
        graph_tensors = torch.load(data_path)
        # make the graph
        graph = Data(**graph_tensors)
        # apply the functions specified in pre_filter and pre_transform
        data_list = [graph]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  

if __name__ == "__main__":
    # Load the connectome data
    dataset = CElegansDataset()
    connectome = dataset[0]
    # Investigate the C. elegans connectome graph
    print()
    print("C. elegans connectome graph loaded successfully!")
    print("Attributes:", "\n", connectome.keys, "\n",
    f"Num. nodes {connectome.num_nodes}, Num. edges {connectome.num_edges}, " 
    f"Num. node features {connectome.num_node_features}", end="\n")
    print(f"\tHas isolated nodes: {connectome.has_isolated_nodes()}")
    print(f"\tHas self-loops: {connectome.has_self_loops()}")
    print(f"\tIs undirected: {connectome.is_undirected()}")
    print(f"\tIs directed: {connectome.is_directed()}") 
    print()