from data._pkg import *


# Init logger
logger = logging.getLogger(__name__)


class NeuralActivityDataset(torch.utils.data.Dataset):
    """A custom neural activity time-series prediction dataset.

    Using NeuralActivityDataset will ensure that sequences are generated
    in a principled and deterministic way, and that every sample
    generated is unique. A map-style dataset implements the `__getitem__()`
    and `__len__()` protocols, and represents a map from indices/keys to
    data samples. Accesing with `dataset[idx]` reads the `idx`-th time-series
    and the corresponding target from memory.

    Notes
    -----
    * The generated samples can have overlapped time steps
    """

    def __init__(
        self,
        data,
        time_vec,
        neurons_mask,
        wormID,
        worm_dataset,
        seq_len=1,
        num_samples=100,
        reverse=False,
        use_residual=False,
    ):
        """Initializes a new instance of the NeuralActivityDataset.

        Parameters
        ----------
        data : torch.tensor
            Data with shape (max_timesteps, num_neurons).
        seq_len : int, default=1
            Sequences of length `seq_len` are generated until the dataset
            size is achieved.
        num_samples : int, default=100
            Total number of (input, target) data pairs to generate.
            0 < num_samples <= max_timesteps
        neurons : None, int or array-like, default=None
            Index of neuron(s) to return data for. Returns data for all
            neurons if None.
        time_vec : None or array-like, default=None
            A vector of the time (in seconds) corresponding to the time
            axis (axis=0) of the `data` tensor.
        reverse : bool, default=False
            Whether to sample sequences backward from end of the data.

        Returns
        -------
        (X, Y, metadata) : tuple
            Batch of data samples, where
        X : torch.tensor
            Input tensor with shape (batch_size, seq_len, num_neurons)
        Y: torch.tensor
            Target tensor with same shape as X
        metadata : dict
            Metadata information about samples.
            keys: 'seq_len', 'start' index , 'end' index
        """

        super().__init__()

        # Check the inputs
        assert torch.is_tensor(data), "Recast the data as type `torch.tensor`."
        assert data.ndim == 2, "Reshape the data tensor as (time, neurons)"
        assert isinstance(seq_len, int) and 0 < seq_len <= data.size(
            0
        ), "Enter an integer sequence length 0 < `seq_len` <= max_timesteps."

        # Create time vector if not provided
        assert torch.is_tensor(
            time_vec
        ), "Recast the time vector as type `torch.tensor`."
        assert time_vec.squeeze().ndim == 1 and len(time_vec) == data.size(
            0
        ), "Time vector must have shape (len(data), )"
        self.time_vec = time_vec.squeeze()

        self.max_timesteps, self.num_neurons = data.shape
        self.wormID = wormID
        self.worm_dataset = worm_dataset
        self.seq_len = seq_len
        self.reverse = reverse
        self.use_residual = use_residual
        self.unique_time_steps = set()

        self.data = data
        self.neurons_mask = neurons_mask
        self.num_samples = num_samples
        self.data_samples = self.__data_generator()

        assert self.num_samples == len(
            self.data_samples
        ), "Wrong number of sequences generated!"

    def __len__(self):
        """Denotes the total number of samples."""
        return self.num_samples

    def __getitem__(self, index):
        """Generates one sample of data."""
        return self.data_samples[index]

    def parfor_func(self, start):
        """Helper function for parallelizing `__data_generator`.

        This function is applied to each `start` index in the
        `__data_generator` method.
        """

        # Define an end index
        end = start + self.seq_len
        # Get the time vector
        time_vec = self.time_vec[start:end].detach().clone()
        # Calculate the average dt
        avg_dt = torch.diff(time_vec).mean()

        # Data samples: input (X) and target (Y)
        X = self.data[start:end, :].detach().clone()
        Y = self.data[start + 1 : end + 1, :].detach().clone()  # Overlap

        # Calculate the residual (forward first derivative)
        Res = (Y - X).detach() / avg_dt

        metadata = dict(
            wormID=self.wormID,
            worm_dataset=self.worm_dataset,
            time_vec=time_vec,
            start_time=start,
        )

        # Return sample
        if self.use_residual:
            Y = Res

        return X, Y, self.neurons_mask, metadata

    def __data_generator(self):
        """Private method for generating data samples.

        This function split the data into sequences of length `seq_len`
        by calling `parfor_func` in parallel.

        Notes
        -----
        * The samples can have overlapped time steps

        """

        # Define length of time
        T = self.max_timesteps
        # Dataset will contain sequences of length `seq_len`
        L = self.seq_len

        # All start indices
        start_range = (
            np.linspace(0, T - L - 1, self.num_samples, dtype=int)
            if not self.reverse  # generate from start to end
            else np.linspace(  # generate from end to start
                T - L - 1, 0, self.num_samples, dtype=int
            )
        )
        # Sequential processing (applying the function to each element)
        data_samples = list(map(self.parfor_func, start_range))

        # update the unique_time_steps set
        for _, _, _, metadata in data_samples:
            time_steps = metadata["time_vec"].numpy()  # assuming it's a torch tensor
            self.unique_time_steps.update(time_steps)

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


def select_named_neurons(multi_worm_dataset, num_named_neurons):
    """Select the `num_named_neurons` neurons from the dataset.

    Parameters
    ----------
    multi_worm_dataset : dict
        Multi-worm dataset to select neurons from.

    Returns
    -------
    multi_worm_dataset : dict
        Multi-worm dataset with selected neurons.
    """

    worms_to_drop = []

    for wormID, data in multi_worm_dataset.items():
        # Check if worm has named neurons
        if torch.sum(data["named_neurons_mask"]) == 0:
            worms_to_drop.append(wormID)
            continue

        # Skip if num_named_neurons is 'all'
        if num_named_neurons == "all":
            continue

        # Verify if new num_named_neurons <= actual num_named_neurons
        if num_named_neurons > data["num_named_neurons"]:
            worms_to_drop.append(wormID)
            continue

        else:
            # Overwrite the neuron values
            data["num_named_neurons"] = num_named_neurons
            data["num_unknown_neurons"] = data["num_neurons"] - num_named_neurons

            # Select the neurons to keep
            named_neurons_mask = data["named_neurons_mask"]
            neurons_to_keep = np.random.choice(
                np.where(named_neurons_mask == True)[0],
                num_named_neurons,
                replace=False,
            )

            # Overwrite the named neuron masks
            named_neurons_mask = torch.zeros_like(named_neurons_mask)
            named_neurons_mask[neurons_to_keep] = True

            # Overwrite the unknown neuron masks
            unknown_neurons_mask = data["unknown_neurons_mask"]
            unknown_neurons_mask = ~(named_neurons_mask ^ unknown_neurons_mask)

            slot_to_named_neuron = data["slot_to_named_neuron"]
            slot_to_unknown_neuron = data["slot_to_unknown_neuron"]

            # New unknown neurons
            new_unknown_neurons_map = {
                slot: named_neuron
                for slot, named_neuron in slot_to_named_neuron.items()
                if slot not in neurons_to_keep
            }
            slot_to_unknown_neuron.update(new_unknown_neurons_map)

            # New named neurons
            slot_to_named_neuron = {
                slot: named_neuron
                for slot, named_neuron in slot_to_named_neuron.items()
                if slot in neurons_to_keep
            }

            # Invert new mappings
            named_neuron_to_slot = {
                named_neuron: slot
                for slot, named_neuron in slot_to_named_neuron.items()
            }
            unknown_neuron_to_slot = {
                unknown_neuron: slot
                for slot, unknown_neuron in slot_to_unknown_neuron.items()
            }

            # Update the dataset
            data["named_neurons_mask"] = named_neurons_mask
            data["unknown_neurons_mask"] = unknown_neurons_mask
            data["slot_to_named_neuron"] = slot_to_named_neuron
            data["slot_to_unknown_neuron"] = slot_to_unknown_neuron
            data["named_neuron_to_slot"] = named_neuron_to_slot
            data["unknown_neuron_to_slot"] = unknown_neuron_to_slot

    # Drop worms with less than `num_named_neurons` neurons
    if len(worms_to_drop) > 0:
        logger.info(
            "Dropping {} worms from {}. {} remaining.".format(
                len(worms_to_drop),
                data["dataset"],
                len(list(set(multi_worm_dataset.keys()) - set(worms_to_drop))),
            )
        )

    for wormID in worms_to_drop:
        del multi_worm_dataset[wormID]

    return multi_worm_dataset


def rename_worm_keys(d):
    """
    Auxiliar function to rename the keys of a combined dataset.
    """
    # Sort the keys
    sorted_keys = sorted(d.keys(), key=lambda x: int(x.replace("worm", "")))

    # Create a mapping from old keys to new keys
    key_mapping = {old_key: f"worm{i}" for i, old_key in enumerate(sorted_keys)}

    # Return the dictionary with keys renamed
    return {key_mapping[key]: d[key] for key in sorted_keys}


def filter_loaded_combined_dataset(combined_dataset, num_worms, num_named_neurons):
    """
    Auxiliar function to filter worms when loading a combined dataset.

    Parameters
    ----------
    combined_dataset : dict
        Multi-worm dataset to filter the worms from.
    num_worms : int
        Number of worms to keep.
    num_named_neurons : int
        Number of named neurons to keep.

    Returns
    -------
    combined_dataset : dict
        Filtered multi-worm dataset.
    """

    combined_dataset = select_named_neurons(combined_dataset, num_named_neurons)

    # Verify if len(combined_dataset) is >= num_worms
    if num_worms != "all":
        assert (
            len(combined_dataset) >= num_worms
        ), "num_worms must be less than or equal to the number of worms in the combined dataset. "

        # Select `num_worms` worms
        wormIDs = [wormID for wormID in combined_dataset.keys()]
        wormIDs_to_keep = np.random.choice(wormIDs, size=num_worms, replace=False)
        logger.info(
            "Selecting {} worms from {} in the combined dataset".format(
                len(wormIDs_to_keep), len(combined_dataset)
            )
        )

        # Remove the worms that are not in `wormIDs_to_keep`
        for wormID in wormIDs:
            if wormID not in wormIDs_to_keep:
                combined_dataset.pop(wormID)

    combined_dataset = rename_worm_keys(combined_dataset)

    # Information about the dataset
    dataset_info = {
        "dataset": [],
        "original_index": [],
        "combined_dataset_index": [],
        "neurons": [],
    }

    for worm, data in combined_dataset.items():
        dataset_info["dataset"].append(data["dataset"])
        dataset_info["original_index"].append(data["original_worm"])
        dataset_info["combined_dataset_index"].append(worm)
        worm_neurons = [neuron for slot, neuron in data["slot_to_named_neuron"].items()]
        dataset_info["neurons"].append(worm_neurons)

    dataset_info = pd.DataFrame(dataset_info)

    return combined_dataset, dataset_info


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
    """Function for getting a single worm dataset.

    Outdated or in development.

    Parameters
    ----------
    dataset : dict
        Multi-worm dataset to select a worm from.
    wormid : str or int
        'worm{i}' or {i} where i indexes the worm.

    Raises
    ------
    AssertionError
        If the dataset is not a valid worm dataset.

    Returns
    -------
    single_worm_dataset : dict
        A single worm dataset.
    """
    avail_worms = set(dataset.keys())  # get the available worms
    if isinstance(wormid, str) and wormid.startswith("worm"):
        wormid = wormid.strip("worm")  # get the worm number
        # Exeption if the worm number is not valid
        assert wormid.isnumeric() and int(wormid) <= len(
            avail_worms
        ), "Choose a worm from: {}".format(avail_worms)
        worm = "worm" + wormid
    else:
        # Exeption if the worm number is not valid
        assert isinstance(wormid, int) and wormid <= len(
            avail_worms
        ), "Choose a worm from: {}".format(avail_worms)
        worm = "worm" + str(wormid)
    single_worm_dataset = dataset[worm]
    return single_worm_dataset


def load_connectome():
    """
    Returns the whole nervous system C. elegans connectome.
    """
    return CElegansConnectome()[0]


def load_dataset(name):
    """Load a specified dataset by name.

    This function takes a dataset name as input, checks whether it is in
    the list of valid datasets, and then loads the dataset using the
    corresponding loader function. The loader function is defined in the
    form 'load_{name}', where '{name}' is replaced by the actual dataset name.

    Parameters
    ----------
    name : str
        The name of the dataset to load. Must be one of the valid dataset names.

    Calls
    -----
    load_{dataset} : function in data/_utils.py
        Where dataset = {Kato2015, Nichols2017, Nguyen2017, Skora2018,
                         Kaplan2020, Uzel2022, Flavell2023, Leifer2023} | {Synthetic0000}

    Returns
    -------
    loader():
        The loaded dataset.
    """

    assert (name in VALID_DATASETS) or (
        name in SYNTHETIC_DATASETS
    ), "Unrecognized dataset! Please pick one from:\n{}".format(
        list(VALID_DATASETS | SYNTHETIC_DATASETS)
    )
    loader = eval("load_" + name)  # call the "load" functions below

    return loader()


def load_Custom():
    """
    Loads the last used dataset.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "custom.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    custom_dataset = pickle.load(pickle_in)
    return custom_dataset


def load_Synthetic0000():
    """
    Loads the synthetic dataset Synthetic0000.
    """
    # ensure the data has been created
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Synthetic0000.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Synthetic0000 = pickle.load(pickle_in)
    return Synthetic0000


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
    Loads the worm neural activity datasets from Flavell et al., bioRxiv 2023,
    Brain-wide representations of behavior spanning multiple timescales and states in C. elegans.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Flavell2023.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Flavell2023 = pickle.load(pickle_in)
    return Flavell2023


def load_Leifer2023():
    """
    Loads the worm neural activity datasets from Leifer et al., bioRxiv 2023,
    Neural signal propagation atlas of C. elegans.
    """
    # ensure the data has been preprocessed
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", "Leifer2023.pickle")
    assert os.path.exists(file)
    pickle_in = open(file, "rb")
    # unpickle the data
    Leifer2023 = pickle.load(pickle_in)
    return Leifer2023


def select_desired_worms(multi_worm_dataset, num_worms):
    # If num_worms is 'all', return the whole dataset
    if num_worms == "all":
        return multi_worm_dataset

    wormIDs = [wormID for wormID in multi_worm_dataset.keys()]
    dataset_name = multi_worm_dataset[wormIDs[0]]["dataset"]

    # num_worms can be string, list or int
    if isinstance(num_worms, str):
        # User requested one specific worm
        logger.info("Using {} from {}".format(num_worms, dataset_name))
        wormIDs_to_keep = [num_worms]
    elif isinstance(num_worms, int):
        # User requested a specific number of worms (random pick)
        assert num_worms <= len(
            multi_worm_dataset
        ), f"Chosen number of worms must be less than or equal to the number of worms in {dataset_name}."
        wormIDs_to_keep = np.random.choice(wormIDs, size=num_worms, replace=False)
        logger.info(
            "Using {} worms from {} (random pick)".format(num_worms, dataset_name)
        )
    else:
        # User requested specific worms
        assert len(num_worms) <= len(
            multi_worm_dataset
        ), f"Chosen number of worms must be less than or equal to the number of worms in {dataset_name}."
        wormIDs_to_keep = num_worms
        logger.info("Using {} from {}".format(wormIDs_to_keep, dataset_name))

    # Remove the worms that are not in `wormIDs_to_keep`
    for wormID in wormIDs:
        if wormID not in wormIDs_to_keep:
            multi_worm_dataset.pop(wormID)

    return multi_worm_dataset


def create_combined_dataset(experimental_datasets, num_named_neurons):
    """Returns a dict with the worm data of all requested datasets.

    Parameters
    ----------
    config: DictConfig
        Hydra configuration object.

    Calls
    -----
    load_dataset : function in data/_utils.py
        Load a specified dataset by name.

    Returns
    -------
    dataset: dict
        Dictionary of single worm data objects.

    Notes
    -----
    * The keys of the dictionary are the worm IDs ('worm0', 'worm1', etc.).
    * The features of each worm are (22 in total):
        'calcium_data', 'dataset', 'dt', 'max_timesteps', 'named_neuron_to_slot',
        'named_neurons_mask', 'neuron_to_slot', 'neurons_mask',
        'num_named_neurons', 'num_neurons', 'num_unknown_neurons',
        'residual_calcium', 'slot_to_named_neuron', 'slot_to_neuron',
        'slot_to_unknown_neuron', 'smooth_calcium_data', 'smooth_method',
        'smooth_residual_calcium', 'time_in_seconds', 'unknown_neuron_to_slot',
        'unknown_neurons_mask', 'worm'.
    """

    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}

    # Load the dataset(s)
    combined_dataset = dict()

    for dataset_name, num_worms in experimental_datasets.items():
        # Skip if not worms requested for this dataset
        if num_worms is None or num_worms == 0:
            logger.info("Skipping worms from {} dataset".format(dataset_name))
            continue

        multi_worms_dataset = load_dataset(dataset_name)

        # Select desired worms from this dataset
        multi_worms_dataset = select_desired_worms(multi_worms_dataset, num_worms)

        # Select the `num_named_neurons` neurons (overwrite the masks)
        multi_worms_dataset = select_named_neurons(
            multi_worms_dataset, num_named_neurons
        )

        for worm in multi_worms_dataset:
            if worm in combined_dataset:
                worm_ = (
                    max([int(key.split("worm")[-1]) for key in combined_dataset.keys()])
                    + 1
                )
                worm_ = "worm" + str(worm_)
                combined_dataset[worm_] = multi_worms_dataset[worm]
                combined_dataset[worm_]["worm"] = worm_
                combined_dataset[worm_]["original_worm"] = worm
            else:
                combined_dataset[worm] = multi_worms_dataset[worm]
                combined_dataset[worm]["original_worm"] = worm

    logger.info("Combined dataset has {} worms".format(len(combined_dataset)))

    combined_dataset = rename_worm_keys(combined_dataset)

    # Information about the dataset
    dataset_info = {
        "dataset": [],
        "original_index": [],
        "combined_dataset_index": [],
        "neurons": [],
        "num_neurons": [],
    }

    for worm, data in combined_dataset.items():
        dataset_info["dataset"].append(data["dataset"])
        dataset_info["original_index"].append(data["original_worm"])
        dataset_info["combined_dataset_index"].append(worm)
        worm_neurons = [neuron for slot, neuron in data["slot_to_named_neuron"].items()]
        dataset_info["neurons"].append(worm_neurons)
        dataset_info["num_neurons"].append(len(worm_neurons))

    dataset_info = pd.DataFrame(dataset_info)

    return combined_dataset, dataset_info


def distribute_samples(data_splits, total_nb_samples):
    """Auxiliary function to distribute the samples across the splits.

    Parameters
    ----------
    data_splits : list
        List of data splits.
    total_nb_samples : int
        Total number of samples to distribute.

    Returns
    -------
    samples_to_take : list
        List of the number of samples to take from each split.
    """

    # Calculate the base number of samples for each split
    base_samples_per_split = total_nb_samples // len(data_splits)
    # Calculate the remainder
    remainder = total_nb_samples % len(data_splits)

    samples_to_take = []

    # Distribute the samples
    for i in range(len(data_splits)):
        if i < remainder:
            samples_to_take.append(base_samples_per_split + 1)
        else:
            samples_to_take.append(base_samples_per_split)

    return samples_to_take


def split_combined_dataset(
    combined_dataset,
    num_train_samples,
    num_val_samples,
    seq_len,
    reverse,
    use_residual,
    smooth_data,
):
    """Creates the training and validation datasets from the combined dataset.

    Also returns the number of unique time steps for each worm and each split.

    Parameters
    ----------
    combined_dataset : dict
        Combined dataset to split.
    num_train_samples : int
        Number of training samples.
    num_val_samples : int
        Number of validation samples.
    seq_len : int
        Length of the sequences.
    reverse : bool
        Whether to reverse the sequences.
    use_residual : bool
        Whether to use the residual data.
    smooth_data : bool
        Whether to use the smoothed data.

    Returns
    -------
    train_dataset : list
        List of training datasets.
    val_dataset : list
        List of validation datasets.
    time_step_info : dict
        Dataframe with the number of unique time steps for each worm and each split.
    """

    # Choose whether to use calcium or residual data
    if use_residual:
        key_data = "residual_calcium"
    else:
        key_data = "calcium_data"

    # Choose whether to use original or smoothed data
    if smooth_data:
        key_data = "smooth_" + key_data
    else:
        key_data = key_data

    # Store the training and validation datasets
    train_dataset = []
    val_dataset = []

    # Store the time steps info
    dataset_info_split = {
        "combined_dataset_index": [],
        "train_time_steps": [],
        "num_train_samples": [],
        "train_seq_len": [],
        "val_time_steps": [],
        "num_val_samples": [],
        "val_seq_len": [],
        "smooth_data": [],
        "use_residual": [],
    }

    # Loop through the worms in the dataset
    for wormID, single_worm_dataset in combined_dataset.items():
        # Extract relevant features from the dataset
        data = single_worm_dataset[key_data]
        neurons_mask = single_worm_dataset["named_neurons_mask"]
        time_vec = single_worm_dataset["time_in_seconds"]
        worm_dataset = single_worm_dataset["dataset"]
        original_wormID = single_worm_dataset["original_worm"]

        # Verifications
        assert isinstance(seq_len, int) and 0 < seq_len < len(
            data
        ), "seq_len must be an integer > 0 and < len(data)"

        # Split the data and the time vector into two halves
        data_splits = np.array_split(data, 2)
        time_vec_splits = np.array_split(time_vec, 2)

        # Separate the splits into training and validation sets
        train_data_splits = data_splits[::2]
        train_time_vec_splits = time_vec_splits[::2]
        val_data_splits = data_splits[1::2]
        val_time_vec_splits = time_vec_splits[1::2]

        # Number of samples in each split
        train_samples_per_split = distribute_samples(
            train_data_splits, num_train_samples
        )
        val_samples_per_split = distribute_samples(val_data_splits, num_val_samples)

        # Number of unique time steps across all samples for each worm and each split
        train_split_time_steps, val_split_time_steps = 0, 0

        # Create a dataset for each split
        for train_split, train_time_split, num_samples_split in zip(
            train_data_splits, train_time_vec_splits, train_samples_per_split
        ):
            train_dataset.append(
                NeuralActivityDataset(
                    data=train_split.detach(),
                    time_vec=train_time_split.detach(),
                    neurons_mask=neurons_mask,
                    wormID=original_wormID,  # worm ID from the original experimental dataset
                    worm_dataset=worm_dataset,  # name of the original experimental dataset the data is from
                    seq_len=seq_len,
                    num_samples=num_samples_split,
                    use_residual=use_residual,
                    reverse=reverse,
                )
            )
            train_split_time_steps += len(train_dataset[-1].unique_time_steps)
            if len(train_dataset[-1].unique_time_steps) > len(train_split):
                raise ValueError("More time steps from samples than in the train split")

        for val_split, val_time_split, num_samples_split in zip(
            val_data_splits, val_time_vec_splits, val_samples_per_split
        ):
            val_dataset.append(
                NeuralActivityDataset(
                    data=val_split.detach(),
                    time_vec=val_time_split.detach(),
                    neurons_mask=neurons_mask,
                    wormID=original_wormID,  # worm ID of the experimental dataset (original)
                    worm_dataset=worm_dataset,  # dataset where the worm comes from
                    seq_len=seq_len,
                    num_samples=num_samples_split,
                    use_residual=use_residual,
                    reverse=reverse,
                )
            )
            val_split_time_steps += len(val_dataset[-1].unique_time_steps)
            if len(val_dataset[-1].unique_time_steps) > len(val_split):
                raise ValueError("More time steps from samples than in the val. split")

        # Store the number of unique time steps for each worm
        dataset_info_split["combined_dataset_index"].append(wormID)

        dataset_info_split["train_time_steps"].append(train_split_time_steps)
        dataset_info_split["train_seq_len"].append(seq_len)
        dataset_info_split["num_train_samples"].append(num_train_samples)

        dataset_info_split["val_time_steps"].append(val_split_time_steps)
        dataset_info_split["val_seq_len"].append(seq_len)
        dataset_info_split["num_val_samples"].append(num_val_samples)

        dataset_info_split["smooth_data"].append(smooth_data)
        dataset_info_split["use_residual"].append(use_residual)

    # Concatenate the datasets
    train_dataset = torch.utils.data.ConcatDataset(
        train_dataset
    )  # Nb of train examples = nb train samples * nb of worms
    val_dataset = torch.utils.data.ConcatDataset(
        val_dataset
    )  # Nb of val examples = nb train samples * nb of worms

    # Convert information about the split datasets into a dataframe
    dataset_info_split = pd.DataFrame(dataset_info_split)

    return train_dataset, val_dataset, dataset_info_split


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
    max_timesteps, num_neurons = dataset.shape
    assert max_timesteps == single_worm_dataset["max_timesteps"]
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
    new_x = torch.zeros(graph.num_nodes, max_timesteps, dtype=torch.float32)
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
