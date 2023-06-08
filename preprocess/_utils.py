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


def total_var_reg_smooth(x, t, alpha=1e-2, order=0):
    """
    Total variational regularization for smoothing a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        alpha (float): The regularization parameter.
        order (int): The order of the total variational derivative.

    Returns:
        x_smooth (ndarray): The smoothed time series.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    t = t.squeeze()
    # Total variational derivative
    # dt = np.diff(t, prepend=t[0] - np.diff(t)[0])
    # dxdt_totalvar = dxdt(x, t, kind="trend_filtered", order=order, alpha=alpha, axis=0)
    # x_smooth = np.cumsum(dxdt_totalvar, axis=0) * dt.reshape(-1, 1)
    ###
    diff_method = derivative.TrendFiltered(alpha=alpha, order=order)
    x_smooth = diff_method.x(x, t, axis=0)
    ###
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def kalman_smooth(x, t, alpha=1):
    """Kalman derivative for smoothing a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        alpha (int): Kernel size for the Kalman filter.

    Returns:
        x_smooth (ndarray): The smoothed time series.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    # Kalman filter derivative
    # t = t.squeeze()
    # dt = np.diff(t, prepend=t[0] - np.diff(t)[0])
    # dxdt_kalman = dxdt(x, t, kind="kalman", alpha=alpha, axis=0)
    # x_smooth = np.cumsum(dxdt_kalman) * dt
    ###
    diff_method = derivative.Kalman(alpha=alpha)
    x_smooth = diff_method.x(x, t, axis=0)
    ###
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def kernel_smooth(x, t, sigma=1, lmbd=0.1, kernel="rbf"):
    """Kernel derivative for smoothing a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        sigma (float): The kernel length-scale parameter.
        lmbd (float): The kernel regularization parameter.
        kernel (str): The kernel type.

    Returns:
        x_smooth (ndarray): The smoothed time series.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    # Kernel derivative
    # t = t.squeeze()
    # dt = np.diff(t, prepend=t[0] - np.diff(t)[0])
    # dxdt_kernel = dxdt(
    #     x, t, kind="kernel", sigma=sigma, lmbd=lmbd, kernel=kernel, axis=0
    # )
    # x_smooth = np.cumsum(dxdt_kernel, axis=0) * dt.reshape(-1, 1)
    ###
    diff_method = derivative.Kernel(sigma=sigma, lmbd=lmbd, kernel=kernel)
    x_smooth = diff_method.x(x, t, axis=0)
    ###
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def fourier_transform_smooth(x, t, percent=0.1):
    """Uses the FFT to smooth a multidimensional time series.

    Smooths a multidimensional time series by keeping the lowest `percent`
    fraction of the maximum frequency from the FFT of the input signal.

    Parameters:
        x (ndarray): The input time series to be smoothed.
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        percent (float): The percentage of max frequency to keep.

    Returns:
        x_smooth (ndarray): The smoothed time series.
    """
    isnumpy = isinstance(x, np.ndarray)
    if isnumpy:
        x = torch.from_numpy(x)
    dim = x.ndim
    if dim == 1:
        x = x.unsqueeze(-1)
    t = t.squeeze()
    dt = np.median(np.diff(t, prepend=t[0] - np.diff(t)[0]))  # sampling time
    x_smooth = torch.zeros_like(x)
    frequencies = torch.fft.rfftfreq(x.shape[0], d=dt)
    threshold = torch.abs(frequencies)[
        int(frequencies.shape[0] * percent)
    ]  # keep lowest `percent` of frequencies
    oneD_kernel = torch.abs(frequencies) < threshold
    fft_input = torch.fft.rfftn(x, dim=0)
    oneD_kernel = oneD_kernel.repeat(x.shape[1], 1).T
    fft_result = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
    x_smooth[0 : fft_result.shape[0]] = fft_result
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if isnumpy:
        x_smooth = x_smooth.cpu().numpy()
    return x_smooth


def smooth_data_preprocess(calcium_data, time_in_seconds, smooth_method):
    """Smooths the calcium data provided as a (time, neurons) array `calcium_data`.

    Returns the denoised signals calcium signals using the method specified by `smooth_method`.

    Args:
        calcium_data: original calcium data from dataset
        time_in_seconds: time vector corresponding to calcium_data
        smooth_method: the method used to smooth the data

    Returns:
        smooth_ca_data: calcium data that is smoothed
    """

    if smooth_method is None:
        smooth_ca_data = calcium_data
    elif str(smooth_method).lower() == "fft":
        smooth_ca_data = fourier_transform_smooth(calcium_data, time_in_seconds)
    elif str(smooth_method).lower() == "ka":
        smooth_ca_data = kalman_smooth(calcium_data, time_in_seconds)
    elif str(smooth_method).lower() == "ke":
        smooth_ca_data = kernel_smooth(calcium_data, time_in_seconds)
    elif str(smooth_method).lower() == "tvr":
        smooth_ca_data = total_var_reg_smooth(calcium_data, time_in_seconds)
    else:
        raise TypeError("Check `config/preprocess.yml` for available smooth methods.")
    return smooth_ca_data


class CalciumDataReshaper:
    def __init__(self, worm_dataset):
        self.worm_dataset = worm_dataset
        self.named_neuron_to_idx = {}
        self.unknown_neuron_to_idx = {}
        self.slot_to_named_neuron = {}
        self.slot_to_unknown_neuron = {}
        self.slot_to_neuron = {}
        self.max_timesteps = self.worm_dataset["max_timesteps"]
        self.dtype = torch.float32

        self._init_neuron_data()
        self._reshape_data()

    def _init_neuron_data(self):
        self.time_in_seconds = self.worm_dataset["time_in_seconds"]
        self.origin_calcium_data = self.worm_dataset["calcium_data"]
        self.smooth_calcium_data = self.worm_dataset["smooth_calcium_data"]
        self.residual_calcium = self.worm_dataset["residual_calcium"]
        self.smooth_residual_calcium = self.worm_dataset["smooth_residual_calcium"]
        self.num_unknown_neurons = self.worm_dataset["num_unknown_neurons"]
        self.neuron_to_idx = self.worm_dataset["neuron_to_idx"]
        self.idx_to_neuron = self.worm_dataset["idx_to_neuron"]

    def _reshape_data(self):
        self._prepare_initial_data()
        self._fill_named_neurons_data()
        self._fill_unknown_neurons_data()
        self._update_worm_dataset()
        self._remove_old_mappings()

    def _prepare_initial_data(self):
        assert (
            len(self.idx_to_neuron) == self.origin_calcium_data.shape[1]
        ), "Number of neurons in calcium dataset does not match number of recorded neurons."
        self.named_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self.unknown_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self._init_empty_calcium_data()
        self._tensor_time_data()

    def _init_empty_calcium_data(self):
        self.standard_calcium_data = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_residual_calcium = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_smooth_calcium_data = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_residual_smooth_calcium = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )

    def _tensor_time_data(self):
        self.time_in_seconds = torch.from_numpy(self.time_in_seconds).to(self.dtype)
        if self.time_in_seconds.ndim == 1:
            self.time_in_seconds = self.time_in_seconds.unsqueeze(-1)
        dt = np.gradient(self.time_in_seconds, axis=0)
        dt[dt == 0] = np.finfo(float).eps
        self.dt = torch.from_numpy(dt).to(self.dtype)
        if self.dt.ndim == 1:
            self.dt = self.dt.unsqueeze(-1)

    def _fill_named_neurons_data(self):
        for slot, neuron in enumerate(NEURONS_302):
            if neuron in self.neuron_to_idx:  # named neuron
                idx = self.neuron_to_idx[neuron]
                self.named_neuron_to_idx[neuron] = idx
                self._fill_calcium_data(idx, slot)
                self.named_neurons_mask[slot] = True
                self.slot_to_named_neuron[slot] = neuron

    def _fill_calcium_data(self, idx, slot):
        self.standard_calcium_data[:, slot] = torch.from_numpy(
            self.origin_calcium_data[:, idx]
        )
        self.standard_residual_calcium[:, slot] = torch.from_numpy(
            self.residual_calcium[:, idx]
        )
        self.standard_smooth_calcium_data[:, slot] = torch.from_numpy(
            self.smooth_calcium_data[:, idx]
        )
        self.standard_residual_smooth_calcium[:, slot] = torch.from_numpy(
            self.smooth_residual_calcium[:, idx]
        )

    def _fill_unknown_neurons_data(self):
        free_slots = list(np.where(~self.named_neurons_mask)[0])
        for neuron in set(self.neuron_to_idx) - set(self.named_neuron_to_idx):
            self.unknown_neuron_to_idx[neuron] = self.neuron_to_idx[neuron]
            slot = np.random.choice(free_slots)
            free_slots.remove(slot)
            self.slot_to_unknown_neuron[slot] = neuron
            self._fill_calcium_data(self.neuron_to_idx[neuron], slot)
            self.unknown_neurons_mask[slot] = True

    def _update_worm_dataset(self):
        self.slot_to_neuron.update(self.slot_to_named_neuron)
        self.slot_to_neuron.update(self.slot_to_unknown_neuron)
        self.worm_dataset.update(
            {
                "calcium_data": self.standard_calcium_data,
                "smooth_calcium_data": self.standard_smooth_calcium_data,
                "residual_calcium": self.standard_residual_calcium,
                "smooth_residual_calcium": self.standard_residual_smooth_calcium,
                "time_in_seconds": self.time_in_seconds,
                "dt": self.dt,
                "named_neurons_mask": self.named_neurons_mask,
                "unknown_neurons_mask": self.unknown_neurons_mask,
                "neurons_mask": self.named_neurons_mask | self.unknown_neurons_mask,
                "named_neuron_to_idx": self.named_neuron_to_idx,
                "idx_to_named_neuron": {
                    v: k for k, v in self.named_neuron_to_idx.items()
                },
                "unknown_neuron_to_idx": self.unknown_neuron_to_idx,
                "idx_to_unknown_neuron": {
                    v: k for k, v in self.unknown_neuron_to_idx.items()
                },
                "slot_to_named_neuron": self.slot_to_named_neuron,
                "named_neuron_to_slot": {
                    v: k for k, v in self.slot_to_named_neuron.items()
                },
                "slot_to_unknown_neuron": self.slot_to_unknown_neuron,
                "unknown_neuron_to_slot": {
                    v: k for k, v in self.slot_to_unknown_neuron.items()
                },
                "slot_to_neuron": self.slot_to_neuron,
                "neuron_to_slot": {v: k for k, v in self.slot_to_neuron.items()},
            }
        )

    def _remove_old_mappings(self):
        keys_to_delete = [key for key in self.worm_dataset if "idx" in key]
        for key in keys_to_delete:
            self.worm_dataset.pop(key, None)


def reshape_calcium_data(worm_dataset):
    """
    Reorganizes calcium data into a standard shape of max_timesteps x 302. It
    also creates neuron masks and mappings of neuron labels to indices in the data.
    Converts the data to torch tensors.

    Args:
        worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.

    Returns:
        dict: The modified worm dataset with restructured calcium data.
    """
    reshaper = CalciumDataReshaper(worm_dataset)
    return reshaper.worm_dataset


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
    transform=StandardScaler(),
    smooth_method="fft",
    resample_dt=None,
):
    """Preprocess and then saves C. elegans neural data to .pickle format.

    This function downloads and extracts the open-source datasets if not found in the
    root directory, preprocesses the neural data using the corresponding DatasetPreprocessor class,
    and then saves it to .pickle format. The processed data is saved in the
    data/processed/neural folder for further use.

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
    {Dataset}Preprocessor : class in preprocess/{dataset}.py

    Returns
    -------
    None
        The function's primary purpose is to preprocess the data and save
        it to disk for future use.
    """
    zip_path = os.path.join(ROOT_DIR, zipfile)
    source_path = os.path.join(ROOT_DIR, zipfile.strip(".zip"))
    processed_path = os.path.join(ROOT_DIR, "data/processed/neural")

    # If .zip not found in the root directory, download the curated
    # open-source worm datasets from the host server
    if not os.path.exists(source_path):
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
            print("Dataset:", dataset, end="\n\n")
            # instantiate the relevant preprocessor class
            preprocessor = eval(dataset + "Preprocessor")(
                transform, smooth_method, resample_dt
            )
            # call its method
            preprocessor.preprocess()

    # ... (re)-Pickle a single dataset
    else:
        assert (
            dataset in VALID_DATASETS
        ), "Invalid dataset requested! Please pick one from:\n{}".format(
            list(VALID_DATASETS)
        )
        print("Dataset:", dataset, end="\n\n")
        # instantiate the relevant preprocessor class
        preprocessor = eval(dataset + "Preprocessor")(
            transform, smooth_method, resample_dt
        )
        # call its method
        preprocessor.preprocess()

    # Delete the downloaded raw datasets
    shutil.rmtree(source_path)

    # Create a file to indicate that the preprocessing was successful
    open(os.path.join(processed_path, ".processed"), "a").close()

    return None


def silence_errors():
    logging.getLogger().setLevel(logging.CRITICAL)


def restore_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    # set matplotlib logging level to WARNING
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)


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
        create_neuron_idx(): Method for extracting a neuron label to index mapping from the raw data.

    Note:
        This class is intended to be subclassed, not directly instantiated.
        Specific datasets should implement their own versions of the `load_data`,
        `extract_data`, `smooth_data`, `resample_data`, `normalize_data` and `save_data` methods.

    Example:
        class SpecificDatasetPreprocessor(BasePreprocessor):
            def load_data(self):
                # Implement dataset-specific loading logic here.

    """

    def __init__(
        self,
        dataset_name,
        transform=StandardScaler(),
        smooth_method="FFT",
        resample_dt=0.1,
    ):
        self.dataset = dataset_name
        self.transform = transform
        self.smooth_method = smooth_method
        self.resample_dt = resample_dt
        self.raw_data_path = os.path.join(ROOT_DIR, "opensource_data")
        self.processed_data_path = os.path.join(ROOT_DIR, "data/processed/neural")

    def smooth_data(self, data, time_in_seconds):
        return smooth_data_preprocess(
            data,
            time_in_seconds,
            self.smooth_method,
        )

    def resample_data(self, time_in_seconds, data):
        return interpolate_data(time_in_seconds, data, target_dt=self.resample_dt)

    def normalize_data(self, data):
        return self.transform.fit_transform(data)

    def save_data(self, data_dict):
        file = os.path.join(self.processed_data_path, f"{self.dataset}.pickle")
        with open(file, "wb") as f:
            pickle.dump(data_dict, f)

    def create_neuron_idx(self, unique_IDs):
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
        neuron_to_idx = {
            nid: (str(nid) if name not in set(NEURONS_302) else name)
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        num_named_neurons = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were labeled with a name
        return neuron_to_idx, num_named_neurons

    def load_data(self):
        raise NotImplementedError()

    def extract_data(self):
        raise NotImplementedError()

    def preprocess(self):
        raise NotImplementedError()


class Skora2018Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Skora2018", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"]
        all_traces = arr["traces"]
        timeVectorSeconds = arr["timeVectorSeconds"]
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        # load and preprocess data
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
                neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)

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
                smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method,
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
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")


class Kato2015Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Kato2015", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"] if "IDs" in arr.keys() else arr["NeuronNames"]
        all_traces = arr["traces"] if "traces" in arr.keys() else arr["deltaFOverF_bc"]
        timeVectorSeconds = (
            arr["timeVectorSeconds"] if "timeVectorSeconds" in arr.keys() else arr["tv"]
        )
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        # load and preprocess data
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
                neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
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
                    calcium_data, original_time_in_seconds
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method,
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
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")


class Nichols2017Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Nichols2017", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
        all_traces = arr["traces"]  # neural activity traces corrected for bleaching
        timeVectorSeconds = arr["timeVectorSeconds"]
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        # load and preprocess data
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
                # unique_IDs = [
                #     (j[-1] if isinstance(j, list) else j) for j in neuron_IDs[i]
                # ]
                unique_IDs = [
                    (self.pick_non_none(j) if isinstance(j, list) else j)
                    for j in neuron_IDs[i]
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
                neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
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
                smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method,
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
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")

    def pick_non_none(self, l):
        """
        Returns the first non-None element in a list, l.
        """
        for i in range(len(l)):
            if l[i] is not None:
                return l[i]
        return None


class Kaplan2020Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Kaplan2020", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        # silence logging during loading:
        # ERROR:root:ERROR: MATLAB type not supported: function_handle_workspace, (uint32)
        silence_errors()
        # load data with mat73
        data = mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))
        # turn on logging again after loading
        restore_logging()
        return data

    def extract_data(self, arr):
        all_IDs = arr["neuron_ID"]
        all_traces = arr["traces_bleach_corrected"]
        timeVectorSeconds = arr["time_vector"]
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        # load and preprocess data
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
                neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)

                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
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
                smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method,
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
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")


class Uzel2022Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Uzel2022", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        return mat73.loadmat(os.path.join(self.raw_data_path, self.dataset, file_name))

    def extract_data(self, arr):
        all_IDs = arr["IDs"]
        all_traces = arr["traces"]
        timeVectorSeconds = arr["tv"]
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        # load and preprocess data
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
                neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)
                time_in_seconds = raw_timeVectorSeconds[i].reshape(
                    raw_timeVectorSeconds[i].shape[0], 1
                )
                time_in_seconds = np.array(time_in_seconds, dtype=np.float32)
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
                smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds
                )
                num_unknown_neurons = int(num_neurons) - num_named_neurons
                worm_dict = {
                    worm: {
                        "dataset": self.dataset,
                        "smooth_method": self.smooth_method,
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
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")


class Leifer2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Leifer2023", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        with open(os.path.join(self.raw_data_path, self.dataset, file_name), "r") as f:
            data = [list(map(float, line.split(" "))) for line in f.readlines()]
        data_array = np.array(data)
        return data_array

    def extract_data(self, data_file, labels_file, time_file):
        real_data = self.load_data(data_file)
        label_list = self.load_labels(labels_file)[: real_data.shape[1]]
        time_in_seconds = self.load_time_vector(time_file)
        # remove columns where all values are NaN
        mask = np.argwhere(~np.isnan(real_data).all(axis=0)).flatten()
        real_data = real_data[:, mask]
        label_list = np.array(label_list)[mask].tolist()
        # impute any remaining NaN values
        imputer = IterativeImputer(random_state=0, keep_empty_features=False)
        if np.isnan(real_data).any():
            real_data = imputer.fit_transform(real_data)
        assert real_data.shape[1] == len(
            label_list
        ), "Data and labels do not match!\n Files: {data_file}, {labels_file}"
        assert (
            real_data.shape[0] == time_in_seconds.shape[0]
        ), "Time vector does not match data!\n Files: {data_file}, {time_file}"
        return real_data, label_list, time_in_seconds

    def create_neuron_idx(self, label_list):
        """Overrides the base class method to handle the Leifer 2023 dataset."""
        neuron_to_idx = dict()
        num_unnamed_neurons = 0
        for j, item in enumerate(label_list):
            previous_list = label_list[:j]
            if not item.isalnum():
                label_list[j] = str(j)
                num_unnamed_neurons += 1
                neuron_to_idx[str(j)] = j
            else:
                if item in NEURONS_302 and item not in previous_list:
                    neuron_to_idx[item] = j
                elif item in NEURONS_302 and item in previous_list:
                    label_list[j] = str(j)
                    num_unnamed_neurons += 1
                    neuron_to_idx[str(j)] = j
                else:
                    if (
                        str(item + "L") in NEURONS_302
                        and str(item + "L") not in previous_list
                    ):
                        label_list[j] = str(item + "L")
                        neuron_to_idx[str(item + "L")] = j
                    elif (
                        str(item + "R") in NEURONS_302
                        and str(item + "R") not in previous_list
                    ):
                        label_list[j] = str(item + "R")
                        neuron_to_idx[str(item + "R")] = j
                    else:
                        label_list[j] = str(j)
                        num_unnamed_neurons += 1
                        neuron_to_idx[str(j)] = j
        num_named_neurons = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were labeled with a name
        assert (
            num_named_neurons == len(label_list) - num_unnamed_neurons
        ), "Incorrect calculation of the numer of named neurons."
        return neuron_to_idx, num_named_neurons

    def str_to_float(self, str_num):
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
            raise TypeError("Float has unknown sign.")
        return float_num

    def load_labels(self, file_path):
        with open(file_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines()]
        return labels

    def load_time_vector(self, file_path):
        with open(file_path, "r") as f:
            timeVectorSeconds = [
                self.str_to_float(line.strip("\n")) for line in f.readlines()
            ]
            timeVectorSeconds = np.array(timeVectorSeconds).reshape(-1, 1)
        return timeVectorSeconds

    def preprocess(self):
        # load and preprocess data
        preprocessed_data = {}
        data_dir = os.path.join(self.raw_data_path, self.dataset)
        files = os.listdir(data_dir)
        num_worms = int(len(files) / 6)  # every worm has 6 txt files
        worm_idx = 0  # Initialize worm index outside file loop

        for i in range(0, num_worms):
            worm = f"worm{str(worm_idx)}"
            worm_idx += 1
            data_file = os.path.join(data_dir, f"{str(i)}_gcamp.txt")
            labels_file = os.path.join(data_dir, f"{str(i)}_labels.txt")
            time_file = os.path.join(data_dir, f"{str(i)}_t.txt")
            real_data, label_list, time_in_seconds = self.extract_data(
                data_file, labels_file, time_file
            )
            if len(label_list) == 0:  # skip worms with no neuron labels
                worm_idx -= 1
                continue
            if len(time_in_seconds) < 1000:  # skip worms with very short recordings
                worm_idx -= 1
                continue
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(label_list)
            calcium_data = self.normalize_data(real_data)
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
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(
                residual_calcium, time_in_seconds
            )
            num_unknown_neurons = int(num_neurons) - num_named_neurons
            worm_dict = {
                worm: {
                    "dataset": self.dataset,
                    "smooth_method": self.smooth_method,
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
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")


class Flavell2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Flavell2023", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        if file_name.endswith(".h5"):
            data = h5py.File(
                os.path.join(self.raw_data_path, self.dataset, file_name), "r"
            )
        elif file_name.endswith(".json"):
            with open(
                os.path.join(self.raw_data_path, self.dataset, file_name), "r"
            ) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
        return data

    def extract_data(self, file_data):
        if isinstance(file_data, h5py.File):
            time_in_seconds = np.array(file_data["timestamp_confocal"], dtype=float)
            time_in_seconds = (
                time_in_seconds - time_in_seconds[0]
            )  # start time at 0.0 seconds
            time_in_seconds = time_in_seconds.reshape((-1, 1))
            calcium_data = np.array(file_data["trace_array_F20"], dtype=float)
            neurons = np.array(file_data["neuropal_label"], dtype=str)
            neurons_copy = []
            for neuron in neurons:
                if neuron.replace("?", "L") not in set(neurons_copy):
                    neurons_copy.append(neuron.replace("?", "L"))
                else:
                    neurons_copy.append(neuron.replace("?", "R"))
            neurons = np.array(neurons_copy)

        elif isinstance(file_data, dict):  # assuming JSON format
            avg_time = (
                file_data["avg_timestep"] * 60
            )  # average time step in seconds (float)
            raw_traces = file_data["trace_array"]  # Raw traces (list)
            max_t = len(raw_traces[0])  # Max time steps (int)
            number_neurons = len(raw_traces)  # Number of neurons (int)
            ids = file_data["labeled"]  # Labels (list)

            time_in_seconds = np.arange(
                0, max_t * avg_time, avg_time
            )  # Time vector in seconds
            time_in_seconds = time_in_seconds.reshape((-1, 1))

            calcium_data = np.zeros((max_t, number_neurons))  # All traces
            for i, trace in enumerate(raw_traces):
                calcium_data[:, i] = trace

            neurons = [str(i) for i in range(number_neurons)]

            for i in ids.keys():
                label = ids[str(i)]["label"]
                neurons[int(i) - 1] = label

            # Treat the '?' labels
            for i in range(number_neurons):
                label = neurons[i]

                if not label.isnumeric():
                    if "?" in label and "??" not in label:
                        # Find the group which the neuron belongs to
                        label_split = label.split("?")[0]
                        # Verify possible labels
                        possible_labels = [
                            neuron_name
                            for neuron_name in NEURONS_302
                            if label_split in neuron_name
                        ]
                        # Exclude possibilities that we already have
                        possible_labels = [
                            neuron_name
                            for neuron_name in possible_labels
                            if neuron_name not in neurons
                        ]
                        # Random pick one of the possibilities
                        print(label_split, possible_labels)
                        neurons[i] = np.random.choice(possible_labels)

            for i in range(number_neurons):
                label = neurons[i]

                if not label.isnumeric():
                    if "??" in label:
                        # Find the group which the neuron belongs to
                        label_split = label.split("?")[0]
                        # Verify possible labels
                        possible_labels = [
                            neuron_name
                            for neuron_name in NEURONS_302
                            if label_split in neuron_name
                        ]
                        # Exclude possibilities that we already have
                        possible_labels = [
                            neuron_name
                            for neuron_name in possible_labels
                            if neuron_name not in neurons
                        ]
                        # Random pick one of the possibilities
                        neurons[i] = np.random.choice(possible_labels)

            neurons = np.array(neurons)

            neurons, unique_indices = np.unique(
                neurons, return_index=True, return_counts=False
            )
            calcium_data = calcium_data[
                :, unique_indices
            ]  # only get data for unique neurons

        else:
            raise ValueError(f"Unsupported data type: {type(file_data)}")

        return time_in_seconds, calcium_data, neurons

    def preprocess(self):
        # load and preprocess data
        preprocessed_data = {}
        for i, file in enumerate(
            os.listdir(os.path.join(self.raw_data_path, self.dataset))
        ):
            if not (file.endswith(".h5") or file.endswith(".json")):
                continue
            worm = "worm" + str(i)
            file_data = self.load_data(file)
            time_in_seconds, calcium_data, neurons = self.extract_data(file_data)

            neuron_to_idx, num_named_neurons = self.create_neuron_idx(neurons)
            calcium_data = self.transform.fit_transform(calcium_data)
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
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(
                residual_calcium, time_in_seconds
            )
            num_unknown_neurons = int(num_neurons) - num_named_neurons

            worm_dict = {
                worm: {
                    "dataset": self.dataset,
                    "smooth_method": self.smooth_method,
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

        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")
