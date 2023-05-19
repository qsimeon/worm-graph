from preprocess._pkg import *


def preprocess_connectome(raw_dir, raw_files):
    """Convert the raw connectome data to a graph tensor.

    This function processes raw connectome data, which includes chemical
    synapses and gap junctions, into a format suitable for use in machine
    learning or graph analysis. It reads the raw data in .csv format,
    processes it to extract relevant information, and creates graph
    tensors that represent the C. elegans connectome. The resulting
    graph tensors are saved in the 'data/processed/connectome' folder
    as 'graph_tensors.pt'.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data
    raw_files : list
        Contain the names of the raw connectome data to preprocess

    Returns
    -------
    None
        This function does not return anything, but it does save the
        graph tensors in the 'data/processed/connectome' folder.

    Notes
    -----
    * A connectome is a comprehensive map of the neural connections within
      an organism's brain or nervous system. It is essentially the wiring
      diagram of the brain, detailing how neurons and their synapses are
      interconnected.
    * The connectome data useed here is from Cook et al., 2019.
      If the raw data isn't found, please download it at this link:
      https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip
      and drop in the data/raw folder.
    """

    # Check if the raw connectome data exists
    # TODO: Automatic download if raw data not found (?)
    assert all([os.path.exists(os.path.join(raw_dir, rf)) for rf in raw_files])

    # List of names of all C. elegans neurons
    neurons_all = set(NEURONS_302)

    # Chemical synapses
    GHermChem_Edges = pd.read_csv(os.path.join(raw_dir, "GHermChem_Edges.csv"))  # edges
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes

    # Gap junctions
    GHermElec_Sym_Edges = pd.read_csv(
        os.path.join(raw_dir, "GHermElec_Sym_Edges.csv")
    )  # edges
    GHermElec_Sym_Nodes = pd.read_csv(
        os.path.join(raw_dir, "GHermElec_Sym_Nodes.csv")
    )  # nodes

    # Neurons involved in gap junctions
    df = GHermElec_Sym_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Ggap_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    # Neurons involved in chemical synapses
    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    # Gap junctions
    df = GHermElec_Sym_Edges
    df["EndNodes_1"] = [
        v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_1"]
    ]
    df["EndNodes_2"] = [
        v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_2"]
    ]
    inds = [
        i
        for i in GHermElec_Sym_Edges.index
        if df.iloc[i]["EndNodes_1"] in set(Ggap_nodes.Name)
        and df.iloc[i]["EndNodes_2"] in set(Ggap_nodes.Name)
    ]  # indices
    Ggap_edges = df.iloc[inds].reset_index(drop=True)

    # Chemical synapses
    df = GHermChem_Edges
    df["EndNodes_1"] = [
        v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_1"]
    ]
    df["EndNodes_2"] = [
        v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_2"]
    ]
    inds = [
        i
        for i in GHermChem_Edges.index
        if df.iloc[i]["EndNodes_1"] in set(Gsyn_nodes.Name)
        and df.iloc[i]["EndNodes_2"] in set(Gsyn_nodes.Name)
    ]  # indices
    Gsyn_edges = df.iloc[inds].reset_index(drop=True)

    # Map neuron names (IDs) to indices
    neuron_to_idx = dict(zip(Gsyn_nodes.Name.values, Gsyn_nodes.index.values))
    idx_to_neuron = dict(zip(Gsyn_nodes.index.values, Gsyn_nodes.Name.values))

    # edge_index for gap junctions
    arr = Ggap_edges[["EndNodes_1", "EndNodes_2"]].values
    ggap_edge_index = torch.empty(*arr.shape, dtype=torch.long)
    for i, row in enumerate(arr):
        ggap_edge_index[i, :] = torch.tensor(
            [neuron_to_idx[x] for x in row], dtype=torch.long
        )
    ggap_edge_index = ggap_edge_index.T  # [2, num_edges]

    # edge_index for chemical synapses
    arr = Gsyn_edges[["EndNodes_1", "EndNodes_2"]].values
    gsyn_edge_index = torch.empty(*arr.shape, dtype=torch.long)
    for i, row in enumerate(arr):
        gsyn_edge_index[i, :] = torch.tensor(
            [neuron_to_idx[x] for x in row], dtype=torch.long
        )
    gsyn_edge_index = gsyn_edge_index.T  # [2, num_edges]

    # edge attributes
    num_edge_features = 2

    # edge_attr for gap junctions
    num_edges = len(Ggap_edges)
    ggap_edge_attr = torch.empty(
        num_edges, num_edge_features, dtype=torch.float
    )  # [num_edges, num_edge_features]
    for i, weight in enumerate(Ggap_edges.Weight.values):
        ggap_edge_attr[i, :] = torch.tensor(
            [weight, 0], dtype=torch.float
        )  # electrical synapse encoded as [1,0]

    # edge_attr for chemical synapses
    num_edges = len(Gsyn_edges)
    gsyn_edge_attr = torch.empty(
        num_edges, num_edge_features, dtype=torch.float
    )  # [num_edges, num_edge_features]
    for i, weight in enumerate(Gsyn_edges.Weight.values):
        gsyn_edge_attr[i, :] = torch.tensor(
            [0, weight], dtype=torch.float
        )  # chemical synapse encoded as [0,1]

    # data.x node feature matrix
    num_nodes = len(Gsyn_nodes)
    num_node_features = 1024

    # Generate random data TODO: inject real data istead !
    x = torch.rand(
        num_nodes, num_node_features, dtype=torch.float
    )  # [num_nodes, num_node_features]

    # data.y target to train against
    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    num_classes = len(le.classes_)
    y = torch.tensor(
        le.transform(Gsyn_nodes.Group.values), dtype=torch.int32
    )  # [num_nodes, 1]

    # Save the mapping of encodings to type of neuron
    codes = np.unique(y)
    types = np.unique(Gsyn_nodes.Group.values)
    node_type = dict(zip(codes, types))

    # Graph for electrical connectivity
    electrical_graph = Data(
        x=x, edge_index=ggap_edge_index, edge_attr=ggap_edge_attr, y=y
    )  # torch_geometric package

    # Graph for chemical connectivity
    chemical_graph = Data(
        x=x, edge_index=gsyn_edge_index, edge_attr=gsyn_edge_attr, y=y
    )  # torch_geometric package

    # Merge electrical and chemical graphs into a single connectome graph
    edge_index = torch.hstack((electrical_graph.edge_index, chemical_graph.edge_index))
    edge_attr = torch.vstack((electrical_graph.edge_attr, chemical_graph.edge_attr))
    edge_index, edge_attr = coalesce(
        edge_index, edge_attr, reduce="add"
    )  # features = [elec_wt, chem_wt]

    assert all(chemical_graph.y == electrical_graph.y), "Node labels not matched!"
    x = chemical_graph.x
    y = chemical_graph.y

    # Basic attributes of PyG Data object
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Some additional attributes to the graph
    neurons_all = list(idx_to_neuron.values())
    df = pd.read_csv(
        os.path.join(raw_dir, "LowResAtlasWithHighResHeadsAndTails.csv"),
        header=None,
        names=["neuron", "x", "y", "z"],
    )
    df = df[df.neuron.isin(neurons_all)]
    valids = set(df.neuron)
    keys = [k for k in idx_to_neuron if idx_to_neuron[k] in valids]
    values = list(df[df.neuron.isin(valids)][["x", "z"]].values)

    # Initialize position dict then replace with atlas coordinates if available
    pos = dict(zip(np.arange(graph.num_nodes), np.zeros(shape=(graph.num_nodes, 2))))
    for k, v in zip(keys, values):
        pos[k] = v

    # Assign each node its global node index
    n_id = torch.arange(graph.num_nodes)

    # Save the tensors to use as raw data in the future.
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "idx_to_neuron": idx_to_neuron,
        "node_type": node_type,
        "n_id": n_id,
    }

    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", "graph_tensors.pt"),
    )

    return None


def total_variation_regularization_smooth(x, t, alpha):
    """
    Total variation regularization for smoothing a multidimensional time series.
    TODO: Way too slow, need to optimize.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector corresponding to the input time series.
        alpha (float): The regularization parameter.

    Returns:
        ndarray: The smoothed time series.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, num_features = x.shape
    t = t.squeeze()
    dt = torch.diff(t)
    A = np.zeros((n - 1, n))
    for i in range(n - 1):
        A[i, i] = -1 / dt[i]
        A[i, i + 1] = 1 / dt[i]

    def objective(g):
        g = g.reshape(n, num_features)
        return np.sum((g - x) ** 2) + alpha * np.sum(np.abs(np.dot(A, g)))

    g0 = np.zeros((n, num_features))
    res = minimize(objective, g0.ravel(), method="L-BFGS-B")
    x_smooth = torch.from_numpy(res.x.reshape(n, num_features))
    return x_smooth


def finite_difference_smooth(x_, t_):
    """Uses the Smoothed Finite Difference derivative from PySINDy for smoothing.

    Parameters:
        x_ (tensor): The input time series to be smoothed (time, neurons).
        t_ (tensor): The time vector corresponding to the input time series.

    Returns:
        tensor: The smoothed time series.
    """
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
        t_ = t_.squeeze().cpu().numpy()
    if x_.ndim == 1:
        x_ = x_.reshape(-1, 1)
        t_ = t_.squeeze()
    n, num_features = x_.shape
    dt = np.diff(t_, prepend=t_[0] - np.diff(t_)[0])
    diff = SmoothedFiniteDifference()
    x_smooth = np.zeros((n, num_features))
    for i in range(num_features):
        dxdt = diff._differentiate(x_[:, i], t_)
        x_smooth[:, i] = np.cumsum(dxdt) * dt
    x_smooth = torch.tensor(x_smooth, dtype=torch.float32)
    return x_smooth


def fast_fourier_transform_smooth(x, dt):
    """Uses the FFT to smooth a multidimensional time series.

    Smooths a multidimensional time series by keeping the lowest
    10% of the frequencies from the FFT of the input signal.

    Parameters:
        x (tensor): The input time series to be smoothed.
        dt (float): The uniform time spacing (in seconds) between individual samples.
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, num_features = x.shape
    x_smooth = torch.zeros_like(x)
    frequencies = torch.fft.rfftfreq(n, d=dt)  # dt: sampling time
    threshold = torch.abs(frequencies)[
        int(frequencies.shape[0] * 0.1)
    ]  # keep first 10% of the frequencies
    oneD_kernel = torch.abs(frequencies) < threshold
    fft_input = torch.fft.rfftn(x, dim=0)
    oneD_kernel = oneD_kernel.repeat(num_features, 1).T
    fft_result = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
    x_smooth[0 : min(fft_result.shape[0], x_smooth.shape[0])] = fft_result
    return x_smooth


def smooth_data_preprocess(calcium_data, time_in_seconds, smooth_method, dt=1.0):
    """
    Smooths the calcium data provided as a (time, num_neurons) array `calcium_data`.

    Returns the denoised signals calcium signals using the method specified by `smooth_method`.

    Args:
        calcium_data: original calcium data from dataset
        time_in_seconds: time vector corresponding to calcium_data
        smooth_method: the way to smooth data
        dt: (required for FFT smooth method) the inter-sample time in seconds

    Returns:
        smooth_ca_data: calcium data that is smoothed
    """

    if str(smooth_method).lower() == "fd" or smooth_method is None:
        smooth_ca_data = finite_difference_smooth(calcium_data, time_in_seconds)
    elif str(smooth_method).lower() == "fft":
        smooth_ca_data = fast_fourier_transform_smooth(calcium_data, dt)
    elif str(smooth_method).lower() == "tvr":
        # regularization parameter `alpha` could be fine-tuned
        smooth_ca_data = total_variation_regularization_smooth(
            calcium_data, time_in_seconds, alpha=0.03
        )
    elif str(smooth_method).lower() == "sg":
        smooth_ca_data = torch.from_numpy(
            savgol_filter(calcium_data, 5, 3, mode="nearest", axis=-1)
        )
    else:
        print("Wrong input! Check the `config/preprocess.yml` for available methods.")
        exit(0)

    return smooth_ca_data


def reshape_calcium_data(single_worm_dataset):
    """
    Modifies the worm dataset to restructure calcium data
    into a standard shape of max_timesteps x 302. Inserts neuron
    masks and mappings of neuron labels to indices in the data.
    """
    # get the calcium data for this worm
    origin_calcium_data = single_worm_dataset["calcium_data"]
    smooth_calcium_data = single_worm_dataset["smooth_calcium_data"]
    residual_calcium = single_worm_dataset["residual_calcium"]
    smooth_residual_calcium = single_worm_dataset["smooth_residual_calcium"]
    # get the number of unidentified tracked neurons
    num_unknown_neurons = single_worm_dataset["num_unknown_neurons"]
    # get the neuron to idx map
    neuron_to_idx = single_worm_dataset["neuron_to_idx"]
    idx_to_neuron = single_worm_dataset["idx_to_neuron"]
    # get the length of the time series
    max_timesteps = single_worm_dataset["max_timesteps"]
    # load names of all 302 neurons
    neurons_302 = NEURONS_302
    # check the calcium data
    assert len(idx_to_neuron) == origin_calcium_data.size(
        1
    ), "Number of neurons in calcium dataset does not match number of recorded neurons."
    # create new maps of neurons to indices
    named_neuron_to_idx = dict()
    unknown_neuron_to_idx = dict()
    # create masks of which neurons have data
    named_neurons_mask = torch.zeros(302, dtype=torch.bool)
    unknown_neurons_mask = torch.zeros(302, dtype=torch.bool)
    # create the new calcium data structure
    # len(residual) = len(data) - 1
    standard_calcium_data = torch.zeros(
        max_timesteps, 302, dtype=origin_calcium_data.dtype
    )
    standard_residual_calcium = torch.zeros(
        max_timesteps, 302, dtype=residual_calcium.dtype
    )
    standard_smooth_calcium_data = torch.zeros(
        max_timesteps, 302, dtype=smooth_calcium_data.dtype
    )
    standard_residual_smooth_calcium = torch.zeros(
        max_timesteps, 302, dtype=smooth_residual_calcium.dtype
    )
    # fill the new calcium data structure with data from named neurons
    slot_to_named_neuron = dict((k, v) for k, v in enumerate(neurons_302))
    for slot, neuron in slot_to_named_neuron.items():
        if neuron in neuron_to_idx:  # named neuron
            idx = neuron_to_idx[neuron]
            named_neuron_to_idx[neuron] = idx
            standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
            standard_residual_calcium[:, slot] = residual_calcium[:, idx]
            standard_smooth_calcium_data[:, slot] = smooth_calcium_data[:, idx]
            standard_residual_smooth_calcium[:, slot] = smooth_residual_calcium[:, idx]
            named_neurons_mask[slot] = True
    # randomly distribute the remaining data from unknown neurons
    for neuron in set(neuron_to_idx) - set(named_neuron_to_idx):
        unknown_neuron_to_idx[neuron] = neuron_to_idx[neuron]
    free_slots = list(np.where(~named_neurons_mask)[0])
    slot_to_unknown_neuron = dict(
        zip(
            np.random.choice(free_slots, num_unknown_neurons, replace=False),
            unknown_neuron_to_idx.keys(),
        )
    )
    for slot, neuron in slot_to_unknown_neuron.items():
        idx = unknown_neuron_to_idx[neuron]
        standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
        standard_residual_calcium[:, slot] = residual_calcium[:, idx]
        standard_smooth_calcium_data[:, slot] = smooth_calcium_data[:, idx]
        standard_residual_smooth_calcium[:, slot] = smooth_residual_calcium[:, idx]
        unknown_neurons_mask[slot] = True
    # combined slot to neuron mapping
    slot_to_neuron = dict()
    slot_to_neuron.update(slot_to_named_neuron)
    slot_to_neuron.update(slot_to_unknown_neuron)
    # modify the worm dataset to with new attributes
    single_worm_dataset.update(
        {
            "calcium_data": standard_calcium_data,
            "smooth_calcium_data": standard_smooth_calcium_data,
            "residual_calcium": standard_residual_calcium,
            "smooth_residual_calcium": standard_residual_smooth_calcium,
            "named_neurons_mask": named_neurons_mask,
            "unknown_neurons_mask": unknown_neurons_mask,
            "neurons_mask": named_neurons_mask | unknown_neurons_mask,
            "named_neuron_to_idx": named_neuron_to_idx,
            "idx_to_named_neuron": dict((v, k) for k, v in named_neuron_to_idx.items()),
            "unknown_neuron_to_idx": unknown_neuron_to_idx,
            "idx_to_unknown_neuron": dict(
                (v, k) for k, v in unknown_neuron_to_idx.items()
            ),
            "slot_to_named_neuron": slot_to_named_neuron,
            "named_neuron_to_slot": dict(
                (v, k) for k, v in slot_to_named_neuron.items()
            ),
            "slot_to_unknown_neuron": slot_to_unknown_neuron,
            "unknown_neuron_to_slot": dict(
                (v, k) for k, v in slot_to_unknown_neuron.items()
            ),
            "slot_to_neuron": slot_to_neuron,
            "neuron_to_slot": dict((v, k) for k, v in slot_to_neuron.items()),
        }
    )
    # delete all original index mappings
    keys_to_delete = [key for key in single_worm_dataset if "idx" in key]
    for key in keys_to_delete:
        single_worm_dataset.pop(key, None)
    # return the dataset for this worm
    return single_worm_dataset


def str_to_float(str_num):
    """
    Change textual scientific notation
    into a floating-point number.
    """
    before_e = float(str_num.split("e")[0])
    sign = str_num.split("e")[1][:1]
    after_e = int(str_num.split("e")[1][1:])

    if sign == "+":
        float_num = before_e * math.pow(10, after_e)
    elif sign == "-":
        float_num = before_e * math.pow(10, -after_e)
    else:
        float_num = None
        print("error: unknown sign")
    return float_num


def interpolate_data(time, data, target_dt):
    """Interpolate data using np.interp.

    This function takes the given time points and corresponding data and
    interpolates them to create new data points with the desired time
    interval.

    Parameters
    ----------
    time : numpy.ndarray
        1D array containing the time points corresponding to the data.
    data : numpy.ndarray
        A 2D array containing the data to be interpolated, with shape
        (time, neurons).
    target_dt : float
        The desired time interval between the interpolated data points.
        If None, no interpolation is performed.

    Returns
    -------
    numpy.ndarray, numpy.ndarray: Two arrays containing the interpolated time points and data.
    """
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data

    # Ensure that time is a 1D array
    time = time.squeeze()

    # Interpolate the data
    target_time_np = np.arange(time.min(), time.max(), target_dt)
    num_neurons = data.shape[1]
    interpolated_data_np = np.zeros((len(target_time_np), num_neurons))

    for i in range(num_neurons):
        interpolated_data_np[:, i] = np.interp(target_time_np, time, data[:, i])

    return target_time_np, interpolated_data_np


def pickle_neural_data(
    url,
    zipfile,
    dataset="all",
    transform=StandardScaler(),  # MinMaxScaler(feature_range=(-1, 1))
    smooth_method="fft",
    resample_dt=None,
):
    """Preprocess and then saves C. elegans neural data to .pickle format.

    This function downloads and extracts the open-source datasets if not found in the
    root directory,  proprocesses the neural data and then saves it to .pickle format.
    The processed data is saved in the data/processed/neural folder for
    further use.

    Parameters
    ----------
    url : str
        Download link to a zip file containing the opensource data in raw form.
    zipfile : str
        The name of the zipfile that is being downloaded.
    dataset : str, optional (default: 'all')
        The name of the dataset(s) to be pickled.
        If None, all datasets are pickled.
    transform : object, optional
        The sklearn transformation to be applied to the data.
    smooth_method : str, optional (default: 'fft')
        The smoothing method to apply to the data;
        options are 'sg', 'fft', or 'tvr'.
    resample_dt : float, optional (default: None)
        The resampling time interval in seconds.
        If None, no resampling is performed.

    Calls
    -----
    pickle_{dataset} : function in preprocess/_utils.py
        Where dataset = {Kato2015, Nichols2017, Nguyen2017, Skora2018,
                         Kaplan2020, Uzel2022, Flavell2023, Leifer2023}

    Returns
    -------
    None
        The function's primary purpose is to preprocess the data and save
        it to disk for future use.

    Notes
    -----
    * If you are having problems downloading the data, you can manually download
      and place the zip file into the root directory of this repository.
    * Every called preprocess sub-routine `pickle_{dataset}` has the following
       computation flow:
        1. Load the raw data from the .mat file
        2. Extract the neural data from the raw data
        3. Smooth the neural data
        4. Resample the neural data
        5. Normalize the neural data
        6. Save the processed data to .pickle format
    """

    print("resample_dt", resample_dt, end="\n\n")
    global source_path, processed_path
    zip_path = os.path.join(ROOT_DIR, zipfile)
    source_path = os.path.join(ROOT_DIR, zipfile.strip(".zip"))
    processed_path = os.path.join(ROOT_DIR, "data/processed/neural")

    # If .zip not found in the root directory, download the curated
    # open-source worm datasets from host server
    if not os.path.exists(source_path):
        # ~1.8GB to download
        download_url(url=url, folder=ROOT_DIR, filename=zipfile)

        # Extract all the datasets ... OR
        if dataset.lower() == "all":
            extract_zip(zip_path, folder=source_path)  # Extract zip file
        # Extract just the requested datasets
        else:
            bash_command = [
                "unzip",
                zip_path,
                "{}/*".format(dataset),
                "-d",
                source_path,
            ]
            std_out = subprocess.run(bash_command, text=True)  # Run the bash command
            print(std_out, end="\n\n")

        os.unlink(zip_path)  # Remove zip file

    # (re)-Pickle all the datasets ... OR
    if dataset is None or dataset.lower() == "all":
        for dataset in VALID_DATASETS:
            if dataset.__contains__("sine"):
                continue  # skip these test datasets
            print("Dataset:", dataset, end="\n\n")
            # call the "pickle" functions in preprocess/_utils.py (functions below)
            pickler = eval("pickle_" + dataset)
            pickler(transform, smooth_method, resample_dt)
    # ... (re)-Pickle a single dataset
    else:
        assert (
            dataset in VALID_DATASETS
        ), "Invalid dataset requested! Please pick one from:\n{}".format(
            list(VALID_DATASETS)
        )
        print("Dataset:", dataset, end="\n\n")
        # call the "pickle" functions in preprocess/_utils.py (functions below)
        pickler = eval("pickle_" + dataset)
        pickler(transform, smooth_method, resample_dt)

    # Delete the downloaded raw datasets
    shutil.rmtree(source_path)  # Files too large to push to GitHub

    # Create a file to indicate that the preprocessing was succesful
    open(os.path.join(processed_path, ".processed"), "a").close()

    return None


class BasePreprocessor:
    """
    This is a base class used for preprocessing different types of neurophysiological datasets.

    The class provides a template for loading, extracting, smoothing, resampling, and
    normalizing neural data, as well as saving the processed data in pickle format.
    Specific datasets can be processed by creating a new class that inherits from this base class
    and overriding the methods as necessary.

    Attributes:
        dataset (str): The specific dataset to be preprocessed.
        raw_data (str): The path to the raw dataset.
        processed_data (str): The path to save the processed data.

    Methods:
        load_data(): Method for loading the raw data.
        extract_data(): Method for extracting the neural data from the raw data.
        smooth_data(): Method for smoothing the neural data.
        resample_data(): Method for resampling the neural data.
        normalize_data(): Method for normalizing the neural data.
        save_data(): Method for saving the processed data to .pickle format.

    Note:
        This class is intended to be subclassed, not directly instantiated.
        Specific datasets should implement their own versions of the `load_data`,
        `extract_data`, `smooth_data`, `resample_data`, `normalize_data` and `save_data` methods.

    Example:
        class SpecificDatasetPreprocessor(BasePreprocessor):
            def load_data(self):
                # Implement dataset-specific loading logic here.

    """

    def __init__(self, dataset_name):
        self.dataset = dataset_name
        self.transform = StandardScaler()
        self.smooth_method = "fft"
        self.resample_dt = 1.0
        self.raw_data_path = os.path.join(ROOT_DIR, "opensource_data")
        self.processed_data_path = os.path.join(ROOT_DIR, "data/processed/neural")

    def smooth_data(self, data, time_in_seconds, dt):
        return smooth_data_preprocess(
            data,
            time_in_seconds,
            self.smooth_method,
            dt=np.median(dt),
        )

    def resample_data(self, time_in_seconds, data):
        return interpolate_data(time_in_seconds, data, target_dt=self.resample_dt)

    def normalize_data(self, data):
        return self.transform.fit_transform(data)

    def save_data(self, data_dict):
        file = os.path.join(self.processed_data_path, f"{self.dataset}.pickle")
        with open(file, "wb") as f:
            pickle.dump(data_dict, f)

    def load_data(self):
        raise NotImplementedError()

    def extract_data(self):
        raise NotImplementedError()

    def preprocess(self):
        raise NotImplementedError()


class Skora2018Preprocessor(BasePreprocessor):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"]
        all_traces = arr["traces"]
        timeVectorSeconds = arr["timeVectorSeconds"]
        return all_IDs, all_traces, timeVectorSeconds

    def _create_neuron_idx(self, unique_IDs):
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(unique_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        return neuron_to_idx

    def preprocess(self):
        preprocessed_data = {}
        worm_idx = 0  # Initialize worm index outside file loop
        for file_name in ["WT_fasted.mat", "WT_starved.mat"]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)

            for i, trace_data in enumerate(traces):
                worm = "worm" + str(worm_idx)  # Use global worm index
                worm_idx += 1  # Increment worm index
                unique_IDs = [
                    (j[0] if isinstance(j, list) else j) for j in neuron_IDs[i]
                ]
                unique_IDs = [
                    (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                    for _, j in enumerate(unique_IDs)
                ]
                _, unique_indices = np.unique(unique_IDs, return_index=True)
                unique_IDs = [unique_IDs[_] for _ in unique_indices]
                trace_data = trace_data[
                    :, unique_indices.astype(int)
                ]  # only get data for unique neurons
                neuron_to_idx = self._create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
                num_named_neurons = len(
                    [k for k in neuron_to_idx.keys() if not k.isnumeric()]
                )  # number of neurons that were ID'd
                calcium_data = self.normalize_data(trace_data)
                dt = np.gradient(time_in_seconds, axis=0)
                dt[dt == 0] = np.finfo(float).eps
                residual_calcium = np.gradient(calcium_data, axis=0) / dt
                original_time_in_seconds = time_in_seconds.copy()
                time_in_seconds, calcium_data = self.resample_data(
                    original_time_in_seconds, calcium_data
                )
                time_in_seconds, residual_calcium = self.resample_data(
                    original_time_in_seconds, residual_calcium
                )
                max_timesteps, num_neurons = calcium_data.shape
                smooth_calcium_data = self.smooth_data(
                    calcium_data, time_in_seconds, dt=np.median(dt)
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds, dt=np.median(dt)
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method.upper(),
                        "worm": worm,
                        "calcium_data": calcium_data,
                        "smooth_calcium_data": smooth_calcium_data,
                        "residual_calcium": residual_calcium,
                        "smooth_residual_calcium": smooth_residual_calcium,
                        "neuron_to_idx": neuron_to_idx,
                        "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                        "max_timesteps": int(max_timesteps),
                        "time_in_seconds": time_in_seconds,
                        "dt": dt,
                        "num_neurons": int(num_neurons),
                        "num_named_neurons": num_named_neurons,
                        "num_unknown_neurons": num_unknown_neurons,
                    }
                }
                preprocessed_data.update(worm_dict)
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!")


class Kato2015Preprocessor(BasePreprocessor):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"] if "IDs" in arr.keys() else arr["NeuronNames"]
        all_traces = arr["traces"] if "traces" in arr.keys() else arr["deltaFOverF_bc"]
        timeVectorSeconds = (
            arr["timeVectorSeconds"] if "timeVectorSeconds" in arr.keys() else arr["tv"]
        )
        return all_IDs, all_traces, timeVectorSeconds

    def _create_neuron_idx(self, unique_IDs):
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(unique_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        return neuron_to_idx

    def preprocess(self):
        preprocessed_data = {}
        worm_idx = 0  # Initialize worm index outside file loop
        for file_name in ["WT_Stim.mat", "WT_NoStim.mat"]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)

            for i, trace_data in enumerate(traces):
                worm = "worm" + str(worm_idx)  # Use global worm index
                worm_idx += 1  # Increment worm index
                unique_IDs = [
                    (j[0] if isinstance(j, list) else j) for j in neuron_IDs[i]
                ]
                unique_IDs = [
                    (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                    for _, j in enumerate(unique_IDs)
                ]
                _, unique_indices = np.unique(unique_IDs, return_index=True)
                unique_IDs = [unique_IDs[_] for _ in unique_indices]
                trace_data = trace_data[
                    :, unique_indices.astype(int)
                ]  # only get data for unique neurons
                neuron_to_idx = self._create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
                num_named_neurons = len(
                    [k for k in neuron_to_idx.keys() if not k.isnumeric()]
                )  # number of neurons that were ID'd
                calcium_data = self.normalize_data(trace_data)
                dt = np.gradient(time_in_seconds, axis=0)
                dt[dt == 0] = np.finfo(float).eps
                residual_calcium = np.gradient(calcium_data, axis=0) / dt
                original_time_in_seconds = time_in_seconds.copy()
                time_in_seconds, calcium_data = self.resample_data(
                    original_time_in_seconds, calcium_data
                )
                time_in_seconds, residual_calcium = self.resample_data(
                    original_time_in_seconds, residual_calcium
                )
                max_timesteps, num_neurons = calcium_data.shape
                smooth_calcium_data = self.smooth_data(
                    calcium_data, time_in_seconds, dt=np.median(dt)
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds, dt=np.median(dt)
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method.upper(),
                        "worm": worm,
                        "calcium_data": calcium_data,
                        "smooth_calcium_data": smooth_calcium_data,
                        "residual_calcium": residual_calcium,
                        "smooth_residual_calcium": smooth_residual_calcium,
                        "neuron_to_idx": neuron_to_idx,
                        "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                        "max_timesteps": int(max_timesteps),
                        "time_in_seconds": time_in_seconds,
                        "dt": dt,
                        "num_neurons": int(num_neurons),
                        "num_named_neurons": num_named_neurons,
                        "num_unknown_neurons": num_unknown_neurons,
                    }
                }
                preprocessed_data.update(worm_dict)
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!")


class Nichols2017Preprocessor(BasePreprocessor):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
        all_traces = arr["traces"]  # neural activity traces corrected for bleaching
        timeVectorSeconds = arr["timeVectorSeconds"]
        return all_IDs, all_traces, timeVectorSeconds

    def _create_neuron_idx(self, unique_IDs):
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(unique_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        return neuron_to_idx

    def preprocess(self):
        preprocessed_data = {}
        worm_idx = 0  # Initialize worm index outside file loop
        for file_name in [
            "n2_let.mat",
            "n2_prelet.mat",
            "npr1_let.mat",
            "npr1_prelet.mat",
        ]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)

            for i, trace_data in enumerate(traces):
                worm = "worm" + str(worm_idx)  # Use global worm index
                worm_idx += 1  # Increment worm index
                unique_IDs = [
                    (j[0] if isinstance(j, list) else j) for j in neuron_IDs[i]
                ]
                unique_IDs = [
                    (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                    for _, j in enumerate(unique_IDs)
                ]
                _, unique_indices = np.unique(unique_IDs, return_index=True)
                unique_IDs = [unique_IDs[_] for _ in unique_indices]
                trace_data = trace_data[
                    :, unique_indices.astype(int)
                ]  # only get data for unique neurons
                neuron_to_idx = self._create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
                num_named_neurons = len(
                    [k for k in neuron_to_idx.keys()]
                )  # number of neurons that were ID'd
                calcium_data = self.normalize_data(trace_data)
                dt = np.gradient(time_in_seconds, axis=0)
                dt[dt == 0] = np.finfo(float).eps
                residual_calcium = np.gradient(calcium_data, axis=0) / dt
                original_time_in_seconds = time_in_seconds.copy()
                time_in_seconds, calcium_data = self.resample_data(
                    original_time_in_seconds, calcium_data
                )
                time_in_seconds, residual_calcium = self.resample_data(
                    original_time_in_seconds, residual_calcium
                )
                max_timesteps, num_neurons = calcium_data.shape
                smooth_calcium_data = self.smooth_data(
                    calcium_data, time_in_seconds, dt=np.median(dt)
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds, dt=np.median(dt)
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method.upper(),
                        "worm": worm,
                        "calcium_data": calcium_data,
                        "smooth_calcium_data": smooth_calcium_data,
                        "residual_calcium": residual_calcium,
                        "smooth_residual_calcium": smooth_residual_calcium,
                        "neuron_to_idx": neuron_to_idx,
                        "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                        "max_timesteps": int(max_timesteps),
                        "time_in_seconds": time_in_seconds,
                        "dt": dt,
                        "num_neurons": int(num_neurons),
                        "num_named_neurons": num_named_neurons,
                        "num_unknown_neurons": num_unknown_neurons,
                    }
                }
                preprocessed_data.update(worm_dict)
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!")


class Kaplan2020Preprocessor(BasePreprocessor):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["neuron_ID"]
        all_traces = arr["traces_bleach_corrected"]
        timeVectorSeconds = arr["time_vector"]
        return all_IDs, all_traces, timeVectorSeconds

    def _create_neuron_idx(self, unique_IDs):
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(unique_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        return neuron_to_idx

    def preprocess(self):
        preprocessed_data = {}
        worm_idx = 0  # Initialize worm index outside file loop
        for file_name in [
            "Neuron2019_Data_MNhisCl_RIShisCl.mat",
            "Neuron2019_Data_RIShisCl.mat",
            "Neuron2019_Data_SMDhisCl_RIShisCl.mat",
        ]:
            data_key = "_".join(
                (file_name.split(".")[0].strip("Neuron2019_Data_"), "Neuron2019")
            )
            raw_data = self.load_data(file_name)[data_key]
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)

            for i, trace_data in enumerate(traces):
                worm = "worm" + str(worm_idx)  # Use global worm index
                worm_idx += 1  # Increment worm index

                unique_IDs = [
                    (j[0] if isinstance(j, list) else j) for j in neuron_IDs[i]
                ]
                unique_IDs = [
                    (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                    for _, j in enumerate(unique_IDs)
                ]
                _, unique_indices = np.unique(unique_IDs, return_index=True)
                unique_IDs = [unique_IDs[_] for _ in unique_indices]
                trace_data = trace_data[:, unique_indices.astype(int)]
                neuron_to_idx = self._create_neuron_idx(unique_IDs)

                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
                num_named_neurons = len([k for k in neuron_to_idx.keys()])

                calcium_data = self.normalize_data(trace_data)
                dt = np.gradient(time_in_seconds, axis=0)
                dt[dt == 0] = np.finfo(float).eps
                residual_calcium = np.gradient(calcium_data, axis=0) / dt

                original_time_in_seconds = time_in_seconds.copy()
                time_in_seconds, calcium_data = self.resample_data(
                    original_time_in_seconds, calcium_data
                )
                time_in_seconds, residual_calcium = self.resample_data(
                    original_time_in_seconds, residual_calcium
                )

                max_timesteps, num_neurons = calcium_data.shape
                smooth_calcium_data = self.smooth_data(
                    calcium_data, time_in_seconds, dt=np.median(dt)
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds, dt=np.median(dt)
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons

                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method.upper(),
                        "worm": worm,
                        "calcium_data": calcium_data,
                        "smooth_calcium_data": smooth_calcium_data,
                        "residual_calcium": residual_calcium,
                        "smooth_residual_calcium": smooth_residual_calcium,
                        "neuron_to_idx": neuron_to_idx,
                        "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                        "max_timesteps": int(max_timesteps),
                        "time_in_seconds": time_in_seconds,
                        "dt": dt,
                        "num_neurons": int(num_neurons),
                        "num_named_neurons": num_named_neurons,
                        "num_unknown_neurons": num_unknown_neurons,
                    }
                }
                preprocessed_data.update(worm_dict)
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!")


class Uzel2022Preprocessor(BasePreprocessor):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"]
        all_traces = arr["traces"]
        timeVectorSeconds = arr["tv"]
        return all_IDs, all_traces, timeVectorSeconds

    def _create_neuron_idx(self, unique_IDs):
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(unique_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        return neuron_to_idx

    def preprocess(self):
        preprocessed_data = {}
        worm_idx = 0  # Initialize worm index outside file loop
        for file_name in ["Uzel_WT.mat"]:
            data_key = "Uzel_WT"
            raw_data = self.load_data(file_name)[data_key]
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)

            for i, trace_data in enumerate(traces):
                worm = "worm" + str(worm_idx)  # Use global worm index
                worm_idx += 1  # Increment worm index

                unique_IDs = [
                    (j[0] if isinstance(j, list) else j) for j in neuron_IDs[i]
                ]
                unique_IDs = [
                    (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                    for _, j in enumerate(unique_IDs)
                ]
                unique_IDs, unique_indices = np.unique(unique_IDs, return_index=True)
                trace_data = trace_data[:, unique_indices]
                neuron_to_idx = self._create_neuron_idx(unique_IDs)

                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
                num_named_neurons = len([k for k in neuron_to_idx.keys()])

                calcium_data = self.normalize_data(trace_data)
                dt = np.gradient(time_in_seconds, axis=0)
                dt[dt == 0] = np.finfo(float).eps
                residual_calcium = np.gradient(calcium_data, axis=0) / dt

                original_time_in_seconds = time_in_seconds.copy()
                time_in_seconds, calcium_data = self.resample_data(
                    original_time_in_seconds, calcium_data
                )
                time_in_seconds, residual_calcium = self.resample_data(
                    original_time_in_seconds, residual_calcium
                )

                max_timesteps, num_neurons = calcium_data.shape
                smooth_calcium_data = self.smooth_data(
                    calcium_data, time_in_seconds, dt=np.median(dt)
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds, dt=np.median(dt)
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons

                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method.upper(),
                        "worm": worm,
                        "calcium_data": calcium_data,
                        "smooth_calcium_data": smooth_calcium_data,
                        "residual_calcium": residual_calcium,
                        "smooth_residual_calcium": smooth_residual_calcium,
                        "neuron_to_idx": neuron_to_idx,
                        "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                        "max_timesteps": int(max_timesteps),
                        "time_in_seconds": time_in_seconds,
                        "dt": dt,
                        "num_neurons": int(num_neurons),
                        "num_named_neurons": num_named_neurons,
                        "num_unknown_neurons": num_unknown_neurons,
                    }
                }
                preprocessed_data.update(worm_dict)
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!")


if __name__ == "__main__":
    import os
    import pickle
    import matplotlib.pyplot as plt
    from preprocess._utils import *
    from sklearn.preprocessing import StandardScaler

    # Preprocess the dataset
    preprocessor = Uzel2022Preprocessor(dataset_name="Uzel2022")
    preprocessor.preprocess()

    # Load data from pickle file
    processed_data_path = (
        "/Users/quileesimeon/GitHub Repos/worm-graph/data/processed/neural"
    )
    with open(os.path.join(processed_data_path, "Uzel2022.pickle"), "rb") as f:
        data = pickle.load(f)

    # Extract data for worm0
    worm0_data = data["worm0"]
    print(data.keys(), end="\n\n")
    print(worm0_data["neuron_to_idx"].keys(), end="\n\n")

    # Extract calcium traces. The number of traces you select will depend on the structure of your data
    calcium_traces = worm0_data[
        "calcium_data"
    ]  # Adjust according to your data structure

    print(f"Calcium traces shape: {calcium_traces.shape}")

    # Plot the first few calcium traces
    for i, trace in enumerate(
        calcium_traces.T[:5]
    ):  # Transpose may be needed depending on your data structure
        plt.plot(worm0_data["time_in_seconds"], trace, label=f"Trace {i+1}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Calcium level")
    plt.title("Calcium traces for worm0")
    plt.legend()
    plt.show()
