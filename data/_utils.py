from data._pkg import *
import matplotlib.pyplot as plt


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super(BatchSampler, self).__init__(data_source)
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(self.data_source)


class NeuralActivityDataset(torch.utils.data.Dataset):
    """
    A custom neural activity time-series prediction dataset.
    Using NeuralActivityDataset will ensure that sequences are generated
    in a principled and deterministic way, and that every sample
    generated is unique. A map-style dataset implements the `__getitem__()`
    and `__len__()` protocols, and represents a map from indices/keys to
    data samples. Accesing with `dataset[idx]` reads the `idx`-th time-series
    and the corresponding target from memory.
    """

    def __init__(
            self,
            D,
            neurons=None,
            tau=1,
            seq_len=17,
            size=1000,
            feature_mask=None,
            reverse=False,
    ):
        """
        Args:
          D: torch.tensor. Data w/ shape (max_time, num_neurons, num_features).
          neurons: None, int or array-like. Index of neuron(s) to return data for.
                  Returns data for all neurons if None.
          tau: int, 0 <= tau < max_time//2. Forward time offset of target sequence.
          size: int, 0 < size <= max_time. Batch size. i.e. Number of (input, target)
                                            data pairs to generate.
          seq_len: int. Sequences of length `seq_len` are generated until the dataset `size`
                    is achieved.
          feature_mask: torch.tensor. What features to select for generating dataset.
          reverse: bool. Whether to sample sequences backward from end of the data.
          smooth: str, Method in data smoothing process.
        Returns:
          (X, Y, meta): tuple. Batch of data samples.
            X: torch.tensor. Input tensor w/ shape (batch_size, seq_len,
                                                  num_neurons, num_features)
            Y: torch.tensor. Target tensor w/ same shape as X
            meta: dict. Metadata information about samples.
                        keys: 'seq_len', 'start' index , 'end' index
        """
        super(NeuralActivityDataset, self).__init__()
        # dataset checking
        assert torch.is_tensor(D), "Recast the dataset as `torch.tensor`."
        if D.ndim == 2:
            D = D.unsqueeze(-1)  # expand to 3D if dataset given is 2D
        assert isinstance(seq_len, int), "Enter an integer sequence length `seq_len`."
        self.seq_len = seq_len
        self.max_time, num_neurons, num_features = D.shape
        self.reverse = reverse
        # feature checking
        if feature_mask is not None:
            assert len(feature_mask) == num_features, (
                    "`feature_mask` must have shape (%s, 1)." % num_features
            )
            assert (
                    feature_mask.sum() > 0
            ), "`feature_mask` must select at least 1 feature."
            self.feature_mask = feature_mask
        else:
            self.feature_mask = torch.tensor([1] + (num_features - 1) * [0]).to(
                torch.bool
            )
        # enforce a constraint on using the neurons or signals as features
        if self.feature_mask.sum() == 1:  # single signal
            if neurons is not None:
                self.neurons = np.array(neurons)  # use the subset of neurons given
            else:  # neurons is None
                self.neurons = np.arange(num_neurons)  # use all the neurons
        else:  # multiple signals
            if neurons is not None:
                assert (
                        np.array(neurons).size == 1
                ), "Only select 1 neuron when using > 1 signals as features."
                self.neurons = np.array(neurons)  # use the single neuron given
            else:
                self.neurons = np.array([0])  # use the first neuron
        self.num_neurons = self.neurons.size
        # number of features equals: number of neurons if one signal; number of signals if multiple
        self.num_features = (
            self.feature_mask.sum() if self.num_neurons == 1 else self.num_neurons
        )
        self.D = D
        assert 0 <= tau < self.max_time // 2, "`tau` must be  0 <= tau < max_time//2"
        self.tau = tau
        self.size = size
        self.counter = 0
        self.data_samples, self.batch_sampler = self.__data_generator()

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data_samples)

    def __getitem__(self, index):
        """Generates one sample of data."""
        return self.data_samples[index]

    def __data_generator(self):
        """
        Private method for generating all possible/requested data samples.
        """
        data_samples = []
        # a batch contains all data of a certain length
        batch_indices = []
        # define length of time
        T = self.max_time
        # iterate over all sequences of length eq_len
        L = self.seq_len
        # iterate over all start indices
        start_range = (
            range(0, T - L - self.tau + 1)
            if not self.reverse  # generate from start to end
            else range(T - L - self.tau, -1, -1)  # generate from end to start
        )
        for start in start_range:
            # define an end index
            end = start + L
            # data samples: input, X_tau and target, Y_tau
            X_tau = self.D[start:end, self.neurons, self.feature_mask]
            Y_tau = self.D[
                    start + self.tau: end + self.tau, self.neurons, self.feature_mask
                    ]
            Res_tau = Y_tau - X_tau
            # store metadata about the sample
            tau = torch.tensor(self.tau)
            meta = {"seq_len": L, "start": start, "end": end, "tau": tau}
            # append to data samples
            # change Y_tau to Res_tau
            data_samples.append((X_tau, Res_tau, meta))
            # append index to batch
            batch_indices.append(self.counter)
            self.counter += 1
            # we only want a number of samples up to self.size
            if self.counter >= self.size:
                break
        batch_indices = torch.tensor(batch_indices)
        # size of dataset
        self.size = self.counter
        # save batch indices as attribute
        self.batch_indices = batch_indices
        # create a batch sampler
        batch_sampler = BatchSampler([self.batch_indices])
        # return samples and batch_sampler
        return data_samples, batch_sampler


class CElegansConnectome(InMemoryDataset):
    def __init__(
            self,
            root=os.path.join(ROOT_DIR, "data"),
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        """
        Defines CElegansConnectome as a subclass of a PyG InMemoryDataset.
        """
        super(CElegansConnectome, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[-1])

    @property
    def raw_file_names(self):
        """List of the raw files needed to proceed."""
        return RAW_FILES

    @property
    def processed_file_names(self):
        """List of the processed files needed to proceed."""
        return ["connectome/graph_tensors.pt", "connectome/data.pt"]

    def download(self):
        """Download the raw zip file if not already retrieved."""
        # dataset adapted from from Cook et al. (2019) SI5
        url = RAW_DATA_URL  # base url
        filename = os.path.join("raw_data.zip")
        folder = os.path.join(self.raw_dir)
        download_url(
            url=url, folder=os.getcwd(), filename=filename
        )  # download zip file
        extract_zip(filename, folder=folder)  # unzip data into raw directory
        os.unlink(filename)  # remove zip file

    def process(self):
        """
        Process the raw files and return the dataset (i.e. the connectome graph).
        """
        # preprocessing necessary
        data_path = os.path.join(self.processed_dir, "connectome", "graph_tensors.pt")
        # create a simple dict for loading the connectome
        if not os.path.exists(data_path):  # fun fast preprocess
            subprocess.run("python -u ../preprocess/_main.py", text=True)
        assert os.path.exists(
            data_path
        ), "Must first call `python -u preprocess/_main.py`"
        # load the raw data
        print("Loading from preprocess...")
        graph_tensors = torch.load(data_path)
        # make the graph
        connectome = Data(**graph_tensors)
        # make a dataset with one Data object
        data_list = [connectome]
        # applied specified transforms and filters
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        # save the dataset
        torch.save((data, slices), self.processed_paths[-1])


def find_reliable_neurons(multi_worm_dataset):
    intersection = set()
    for i, worm in enumerate(multi_worm_dataset):
        single_worm_dataset = pick_worm(multi_worm_dataset, worm)
        neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
        curr_set = set(neuron for neuron in neuron_to_idx if not neuron.isnumeric())
        if i == 0:
            intersection |= curr_set
        else:
            intersection &= curr_set
    intersection = sorted(intersection)
    return intersection


def pick_worm(dataset, wormid):
    """
    Function for getting a single worm dataset.
    dataset: str or dict worm dataset to select a worm from.
    wormid: str or int, 'worm{i}' or {i} where i indexes the worm.
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    else:
        assert (
                isinstance(dataset, dict)
                and ("name" in dataset.keys())
                and ("worm0" in set(dataset["generator"]))
        ), "Not a valid worm datset!"
    avail_worms = set(dataset["generator"])
    if isinstance(wormid, str) and wormid.startswith("worm"):
        wormid = wormid.strip("worm")
        assert wormid.isnumeric() and int(wormid) <= len(
            avail_worms
        ), "Choose a worm from: {}".format(avail_worms)
        worm = "worm" + wormid
    else:
        assert isinstance(worm, int) and worm <= len(
            avail_worms
        ), "Choose a worm from: {}".format(avail_worms)
        worm = "worm" + str(wormid)
    single_worm_dataset = dict(dataset["generator"])[worm]
    return single_worm_dataset


def load_connectome():
    """
    Returns the whole nervous system C. elegans connectome.
    """
    return CElegansConnectome()[0]


def load_dataset(name):
    """
    Loads the dataset with the specified name.
    """
    assert (
            name in VALID_DATASETS
    ), "Unrecognized dataset! Please pick one from:\n{}".format(list(VALID_DATASETS))
    loader = eval("load_" + name)
    return loader()


def load_Kato2015():
    """
    Loads the worm neural activity datasets from Kato et al., Cell Reports 2015,
    Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Kato2015.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Kato2015 = pickle.load(pickle_in)
    return Kato2015


def load_Nichols2017():
    """
    Loads the worm neural activity datasets from Nichols et al., Science 2017,
    A global brain state underlies C. elegans sleep behavior.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Nichols2017.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Nichols2017 = pickle.load(pickle_in)
    return Nichols2017


def load_Nguyen2017():
    """
    Loads the worm neural activity datasets from Nguyen et al., PLOS CompBio 2017,
    Automatically tracking neurons in a moving and deforming brain.
    """
    # ensure the data has been preprocessedn
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Nguyen2017.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Nguyen2017 = pickle.load(pickle_in)
    return Nguyen2017


def load_Skora2018():
    """
    Loads the worm neural activity datasets from Skora et al., Cell Reports 2018,
    Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Skora2018.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Skora2018 = pickle.load(pickle_in)
    return Skora2018


def load_Kaplan2020():
    """
    Loads the worm neural activity datasets from Kaplan et al., Neuron 2020,
    Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Kaplan2020.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Kaplan2020 = pickle.load(pickle_in)
    return Kaplan2020


def load_Uzel2022():
    """
    Loads the worm neural activity datasets from Uzel et al 2022., Cell CurrBio 2022,
    A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Uzel2022.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Uzel2022 = pickle.load(pickle_in)
    return Uzel2022


def load_Flavell2023():
    """
    Loads the worm neural activity datasets from from Flavell et al., bioRxiv 2023,
    Brain-wide representations of behavior spanning multiple timescales and states in C. elegans.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Flavell2023.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Flavell2023 = pickle.load(pickle_in)
    return Flavell2023


def graph_inject_data(single_worm_dataset, connectome_graph):
    """
    Find the nodes on the connecotme corresponding to labelled
    neurons in the provided single worm dataset and place the data
    on the connectome graph.
    Returns the full graph with 0s on unlabelled neurons,
    the subgraph with only labelled neurons, the subgraph mask.
    """
    calcium_data = single_worm_dataset["data"]
    graph = connectome_graph
    # get the calcium data for this worm
    dataset = calcium_data.squeeze()
    max_time, num_neurons = dataset.shape
    assert max_time == single_worm_dataset["max_time"]
    assert num_neurons == single_worm_dataset["num_neurons"]
    print("How much real data do we have?", dataset.shape)  # (time, neurons)
    print(
        "Current data on connectome graph:", graph.x.cpu().numpy().shape
    )  # (neurons, time)
    # find the graph nodes matching the neurons in the dataset
    neuron_id = single_worm_dataset["neuron_id"]
    id_neuron = dict((v, k) for k, v in id_neuron.items())
    graph_inds = [
        k for k, v in graph.id_neuron.items() if v in set(id_neuron.values())
    ]  # neuron indices in connectome
    data_inds = [
        k_ for k_, v_ in id_neuron.items() if v_ in set(graph.id_neuron.values())
    ]  # neuron indices in sparse dataset
    # 'inject' the data by creating a clone graph with the desired features
    new_x = torch.zeros(graph.num_nodes, max_time, dtype=torch.float64)
    new_x[graph_inds, :] = dataset[:, data_inds].T
    graph = Data(
        x=new_x,
        y=graph.y,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        node_type=graph.node_type,
        pos=graph.pos,
        num_classes=graph.num_classes,
        id_neuron=graph.id_neuron,
    )
    # assign each node its global node index
    graph.n_id = torch.arange(graph.num_nodes)
    # create the subgraph that has labelled neurons and data
    subgraph_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    subgraph_mask.index_fill_(0, torch.tensor(graph_inds).long(), 1).bool()
    # extract out the subgraph
    subgraph = graph.subgraph(subgraph_mask)
    # reset neuron indices for labeling
    subgraph.id_neuron = {
        i: graph.id_neuron[k] for i, k in enumerate(subgraph.n_id.cpu().numpy())
    }
    subgraph.pos = {i: graph.pos[k] for i, k in enumerate(subgraph.n_id.cpu().numpy())}
    # check out the new attributes
    print(
        "Attributes:",
        "\n",
        subgraph.keys,
        "\n",
        f"Num. nodes {subgraph.num_nodes}, Num. edges {subgraph.num_edges}, "
        f"Num. node features {subgraph.num_node_features}",
        end="\n",
    )
    print(f"\tHas isolated nodes: {subgraph.has_isolated_nodes()}")
    print(f"\tHas self-loops: {subgraph.has_self_loops()}")
    print(f"\tIs undirected: {subgraph.is_undirected()}")
    print(f"\tIs directed: {subgraph.is_directed()}")
    # return the graph, subgraph and mask
    return graph, subgraph, subgraph_mask
