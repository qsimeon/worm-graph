from data._pkg import *


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
        data,
        seq_len=17,
        num_samples=1024,
        neurons=None,
        time_vec=None,
        reverse=False,
        tau=1,
        use_residual=False,
    ):
        """
        Args:
          data: torch.tensor. Data w/ shape (max_time, num_neurons).
          seq_len: int. Sequences of length `seq_len` are generated until the dataset `size`
                    is achieved.
          num_samples: int, 0 < num_samples <= max_time. Total number of (input, target)
                      data pairs to generate.
          neurons: None, int or array-like. Index of neuron(s) to return data for.
                  Returns data for all neurons if None.
          time_vec: None or array-like. A vector of the time (in seconds) corresponding
                    to the time axis (axis 0) of the `data` tensor.
          reverse: bool. Whether to sample sequences backward from end of the data.
          tau: int, 0 < tau < max_time//2. The number of timesteps to the right by which the
                target sequence is offset from input sequence. Deprecated (unused) argument.
        Returns:
            (X, Y, metadata): tuple. Batch of data samples.
            X: torch.tensor. Input tensor w/ shape (batch_size, seq_len,
                                                  num_neurons)
            Y: torch.tensor. Target tensor w/ same shape as X
            metadata: dict. Metadata information about samples.
                        keys: 'seq_len', 'start' index , 'end' index
        """
        super(NeuralActivityDataset, self).__init__()
        # check the inputs
        assert torch.is_tensor(data), "Recast the data as type `torch.tensor`."
        assert data.ndim == 2, "Reshape the data tensor as (time, neurons)"
        assert isinstance(seq_len, int) and 0 < seq_len <= data.size(
            0
        ), "Enter an integer sequence length 0 < `seq_len` <= max_time."
        assert (
            isinstance(tau, int) and 0 <= tau < data.size(0) // 2
        ), "Enter a integer offset `0 <= tau < max_time // 2`."
        # create time vector
        if time_vec is not None:
            assert torch.is_tensor(
                time_vec
            ), "Recast the time vector as type `torch.tensor`."
            assert time_vec.squeeze().ndim == 1 and len(time_vec) == data.size(
                0
            ), "Time vector must have shape (len(data), )"
            self.time_vec = time_vec.squeeze()
        else:
            self.time_vec = torch.arange(data.size(0)).double()
        self.max_time, num_neurons = data.shape
        self.seq_len = seq_len
        self.reverse = reverse
        self.tau = tau
        self.use_residual = use_residual
        # select out requested neurons
        if neurons is not None:
            self.neurons = np.array(neurons)  # use the subset of neurons given
        else:  # neurons is None
            self.neurons = np.arange(num_neurons)  # use all the neurons
        self.num_neurons = self.neurons.size
        self.data = data
        self.num_samples = num_samples
        self.data_samples = self.__data_generator()

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data_samples)

    def __getitem__(self, index):
        """Generates one sample of data."""
        return self.data_samples[index]

    def parfor_func(self, start):
        """
        Helper function for parallelizing `__data_generator`.
        """
        # define an end index
        end = start + self.seq_len
        # get the time vector
        time_vec = self.time_vec[start:end].detach().clone()
        # calculate the average dt
        avg_dt = torch.diff(time_vec).mean()
        # data samples: input (X_tau) and target (Y_tau)
        X_tau = self.data[start:end, self.neurons].detach().clone()
        Y_tau = (
            self.data[start + self.tau : end + self.tau, self.neurons].detach().clone()
        )  # overlap
        # calculate the residual (forward first derivative)
        Res_tau = (Y_tau - X_tau).detach() / (avg_dt * max(1, self.tau))
        # store metadata about the sample
        input = "calcium"
        target = "residual" if self.use_residual else "calcium"
        metadata = dict(
            seq_len=self.seq_len,
            start=start,
            end=end,
            tau=self.tau,
            avg_dt=avg_dt,
            input=input,
            target=target,
            time_vec=time_vec,
        )
        # return sample
        if self.use_residual:
            Y_tau = Res_tau
        return X_tau, Y_tau, metadata

    def __data_generator(self):
        """
        Private method for generating data samples.
        """
        # define length of time
        T = self.max_time
        # dataset will contain sequences of length `seq_len`
        L = self.seq_len
        # all start indices
        start_range = (
            np.linspace(0, T - L - self.tau, self.num_samples, dtype=int)
            if not self.reverse  # generate from start to end
            else np.linspace(  # generate from end to start
                T - L - self.tau, 0, self.num_samples, dtype=int
            )
        )  # overlapping windows
        # sequential processing
        data_samples = list(map(self.parfor_func, start_range))
        return data_samples


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
        return ["connectome/graph_tensors.pt"]

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


def load_sine():
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sine.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sine_noise():
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sine_noise.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sum_sine():
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sum_sine.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sum_sine_noise():
    file = os.path.join(
        ROOT_DIR, "data", "processed", "neural", "sum_sine_noise.pickle"
    )
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sine():
    """
    A test dataset to evaluate models on.
    Independent sinusoid signals.
    Varied frequency and amplitutde.
    """
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sine.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sine_seq():
    """
    A test dataset to evaluate models on.
    Correlated sinusoid signals.
    Varied frequency and amplitutde.
    """
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sine_seq.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sine_noise():
    """
    A test dataset to evaluate models on.
    Independent sinusoid signals + Gaussian noise.
    Varied frequency, phase and noise variance.
    """
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sine_noise.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sine_seq_noise():
    """
    A test dataset to evaluate models on.
    Correlated sinusoid signals + Gaussian noise.
    Varied frequency, phase and noise variance.
    """
    file = os.path.join(
        ROOT_DIR, "data", "processed", "neural", "sine_seq_noise.pickle"
    )
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sum_sine():
    """
    A test dataset to evaluate models on.
    Independent sums-of-sinusoid signals.
    Varied frequency, phase and number of terms in sum.
    """
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "sum_sine.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


def load_sum_sine_noise():
    """
    A test dataset to evaluate models on.
    Independent sums-of-sinusoid signals + Gaussian noise.
    Varied frequency, phase, number of terms in sum and noise variance.
    """
    file = os.path.join(
        ROOT_DIR, "data", "processed", "neural", "sum_sine_noise.pickle"
    )
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    dataset = pickle.load(pickle_in)
    return dataset


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
