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
    """Smooths the calcium data provided as a (time, num_neurons) array `calcium_data`.

    Returns the denoised signals calcium signals using the method specified by `smooth_method`.
    TODO: Make this function only return the smmothed calcium data.

    Args:
        calcium_data: original calcium data from dataset
        time_in_seconds: time vector corresponding to calcium_data
        smooth_method: the way to smooth data
        dt: (required for FFT smooth method) the inter-sample time in seconds

    Returns:
        smooth_ca_data: calcium data that is smoothed
        residual: original residual (calculated by calcium_data)
        residual_smooth_ca_data: residual calculated by smoothed calcium data
    """
    # calculate original residual
    residual = torch.zeros_like(calcium_data)
    residual[1:] = calcium_data[1:, :] - calcium_data[:-1, :]
    print("residual shape:", residual.shape)
    if str(smooth_method).lower() == "fd" or smooth_method == None:
        smooth_ca_data = finite_difference_smooth(calcium_data, time_in_seconds)
    elif str(smooth_method).lower() == "fft":
        smooth_ca_data = fast_fourier_transform_smooth(calcium_data, dt)
    elif str(smooth_method).lower() == "tvr":
        # regularization parameter `alpha`` could be fine-tunded
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
    # calculate residual using smoothed calcium data
    residual_smooth_ca_data = torch.zeros_like(residual)
    residual_smooth_ca_data[1:, :] = smooth_ca_data[1:, :] - smooth_ca_data[:-1, :]
    return smooth_ca_data, residual, residual_smooth_ca_data


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
    """Interpolate data using np.interp, with support for torch.Tensor.

    This function takes the given time points and corresponding data and
    interpolates them to create new data points with the desired time
    interval. The input tensors are first converted to NumPy arrays for
    interpolation, and the interpolated data and time points are then
    converted back to torch.Tensor objects before being returned.

    Parameters
    ----------
    time : torch.Tensor
        1D tensor containing the time points corresponding to the data.
    data : torch.Tensor
        A 2D tensor containing the data to be interpolated, with shape
        (time, neurons).
    target_dt : float
        The desired time interval between the interpolated data points.
        If None, no interpolation is performed.

    Returns
    -------
    torch.Tensor, torch.Tensor: Two tensors containing the interpolated time points and data.
    """
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data

    # Convert input tensors to NumPy arrays
    time_np = time.squeeze().numpy()
    data_np = data.numpy()

    # Interpolate the data
    target_time_np = np.arange(time_np.min(), time_np.max(), target_dt)
    num_neurons = data_np.shape[1]
    interpolated_data_np = np.zeros((len(target_time_np), num_neurons))

    for i in range(num_neurons):
        interpolated_data_np[:, i] = np.interp(target_time_np, time_np, data_np[:, i])

    # Convert the interpolated data and time back to torch.Tensor objects
    target_time = torch.from_numpy(target_time_np).to(torch.float32).unsqueeze(-1)
    interpolated_data = torch.from_numpy(interpolated_data_np).to(torch.float32)

    return target_time, interpolated_data


def pickle_neural_data(
    url,
    zipfile,
    dataset="all",
    transform=MinMaxScaler(feature_range=(-1, 1)),
    # transform=StandardScaler(),
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
        The transformation to be applied to the data.
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


def pickle_Kato2015(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Kato et al., Cell Reports 2015,
    Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans.
    """
    data_dict = dict()

    # 'WT_Stim'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Kato2015", "WT_Stim.mat"))["WT_Stim"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        i_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[i]]
        i_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(i_IDs)
        ]
        _, inds = np.unique(
            i_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(i_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict(
            (v, k) for k, v in neuron_to_idx.items()
        )  # map should be neuron -> index
        time_in_seconds = timeVectorSeconds[i].reshape(timeVectorSeconds[i].shape[0], 1)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)  # samples=time, features=neurons
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds).to(torch.float32)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        print("median dt", np.median(dt))
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Kato2015",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'WT_NoStim'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Kato2015", "WT_NoStim.mat"))[
        "WT_NoStim"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr[
        "NeuronNames"
    ]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["deltaFOverF_bc"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["tv"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for ii, real_data in enumerate(all_traces):
        worm = "worm" + str(ii + i + 1)
        ii_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[ii]]
        ii_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(ii_IDs)
        ]
        _, inds = np.unique(
            ii_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        ii_IDs = [ii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(ii_IDs)
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
        time_in_seconds = timeVectorSeconds[ii].reshape(
            timeVectorSeconds[ii].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds).to(torch.float32)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Kato2015",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Kato2015.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kato2015 = pickle.load(pickle_in)
    print(Kato2015.keys(), end="\n\n")


def pickle_Nichols2017(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Nichols et al., Science 2017,
    A global brain state underlies C. elegans sleep behavior.
    """
    data_dict = dict()

    # 'n2_let'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "n2_let.mat"))[
        "n2_let"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        i_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[i]]
        i_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(i_IDs)
        ]
        _, inds = np.unique(
            i_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(i_IDs)
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
        time_in_seconds = timeVectorSeconds[i].reshape(timeVectorSeconds[i].shape[0], 1)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'n2_prelet'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "n2_prelet.mat"))[
        "n2_prelet"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for ii, real_data in enumerate(all_traces):
        worm = "worm" + str(ii + i + 1)
        ii_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[ii]]
        ii_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(ii_IDs)
        ]
        _, inds = np.unique(
            ii_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        ii_IDs = [ii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(ii_IDs)
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
        time_in_seconds = timeVectorSeconds[ii].reshape(
            timeVectorSeconds[ii].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'npr1_let'
    # load the third .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "npr1_let.mat"))[
        "npr1_let"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for iii, real_data in enumerate(all_traces):
        worm = "worm" + str(iii + ii + 1 + i + 1)
        iii_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[iii]]
        iii_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(iii_IDs)
        ]
        _, inds = np.unique(
            iii_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        iii_IDs = [iii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(iii_IDs)
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
        time_in_seconds = timeVectorSeconds[iii].reshape(
            timeVectorSeconds[iii].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'npr1_prelet'
    # load the fourth .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "npr1_prelet.mat"))[
        "npr1_prelet"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for iv, real_data in enumerate(all_traces):
        worm = "worm" + str(iv + iii + 1 + ii + 1 + i + 1)
        iv_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[iv]]
        iv_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(iv_IDs)
        ]
        _, inds = np.unique(
            iv_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        iv_IDs = [iv_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(iv_IDs)
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
        time_in_seconds = timeVectorSeconds[iv].reshape(
            timeVectorSeconds[iv].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Nichols2017.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nichols2017 = pickle.load(pickle_in)
    print(Nichols2017.keys(), end="\n\n")


def pickle_Nguyen2017(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Nguyen et al., PLOS CompBio 2017,
    Automatically tracking neurons in a moving and deforming brain.
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")

    # WORM 0
    # load .mat file for  worm 0
    arr0 = loadmat(
        os.path.join(source_path, "Nguyen2017", "heatData_worm0.mat")
    )  # load .mat file
    print(list(arr0.keys()), end="\n\n")
    # get data for worm 0
    G2 = arr0[
        "G2"
    ]  # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr0[
        "cgIdx"
    ].squeeze()  # ordered indices derived from heirarchically clustering the correlation matrix.
    real_data0 = G2[cgIdx - 1, :].T  # to show organized traces, use Ratio2(cgIdx,:)
    real_data0 = imputer.fit_transform(real_data0)  # impute missing values (i.e. NaNs)
    # time vector
    time_in_seconds0 = arr0.get("hasPointsTime", np.arange(real_data0.shape[0]))
    time_in_seconds0 = time_in_seconds0.reshape(-1, 1)
    time_in_seconds0 = torch.tensor(time_in_seconds0).to(torch.float32)
    num_named0 = 0
    worm0_ID = {i: str(i) for i in range(real_data0.shape[1])}
    worm0_ID = dict((v, k) for k, v in worm0_ID.items())
    # normalize the data
    sc = transform
    real_data0 = sc.fit_transform(real_data0)
    real_data0 = torch.tensor(real_data0, dtype=torch.float32)
    # resample the data to a fixed time step
    time_in_seconds0, real_data0 = interpolate_data(
        time_in_seconds0, real_data0, target_dt=resample_dt
    )
    # calculate the time step
    dt0 = torch.zeros_like(time_in_seconds0)
    dt0[1:] = time_in_seconds0[1:] - time_in_seconds0[:-1]
    # recalculate max_timesteps and num_neurons
    max_time0, num_neurons0 = real_data0.shape
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time0, num_neurons0, num_named0),
        end="\n\n",
    )

    # WORM 1
    # load .mat file for  worm 1
    arr1 = loadmat(
        os.path.join(source_path, "Nguyen2017", "heatData_worm1.mat")
    )  # load .mat file
    print(list(arr1.keys()), end="\n\n")
    # get data for worm 1
    G2 = arr1[
        "G2"
    ]  # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr1[
        "cgIdx"
    ].squeeze()  # ordered indices derived from heirarchically clustering the correlation matrix.
    real_data1 = G2[cgIdx - 1, :].T  # to show organized traces, use Ratio2(cgIdx,:)
    real_data1 = imputer.fit_transform(real_data1)  # replace NaNs
    # time vector
    time_in_seconds1 = arr1.get("hasPointsTime", np.arange(real_data1.shape[0]))
    time_in_seconds1 = time_in_seconds1.reshape(-1, 1)
    time_in_seconds1 = torch.tensor(time_in_seconds1).to(torch.float32)
    num_named1 = 0
    worm1_ID = {i: str(i) for i in range(real_data1.shape[1])}
    worm1_ID = dict((v, k) for k, v in worm1_ID.items())
    # normalize the data
    sc = transform
    real_data1 = sc.fit_transform(real_data1)
    real_data1 = torch.tensor(real_data1, dtype=torch.float32)
    # resample the data to a fixed time step
    time_in_seconds1, real_data1 = interpolate_data(
        time_in_seconds1, real_data1, target_dt=resample_dt
    )
    # calculate the time step
    dt1 = torch.zeros_like(time_in_seconds1)
    dt1[1:] = time_in_seconds1[1:] - time_in_seconds1[:-1]
    # recalculate max_timesteps and num_neurons
    max_time1, num_neurons1 = real_data1.shape
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time1, num_neurons1, num_named1),
        end="\n\n",
    )

    # WORM 2
    # load .mat file for  worm 1
    arr2 = loadmat(
        os.path.join(source_path, "Nguyen2017", "heatData_worm2.mat")
    )  # load .mat file
    print(list(arr2.keys()), end="\n\n")
    # get data for worm 2
    G2 = arr2[
        "G2"
    ]  # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr2[
        "cgIdx"
    ].squeeze()  # ordered indices derived from heirarchically clustering the correlation matrix.
    real_data2 = G2[cgIdx - 1, :].T  # to show organized traces, use Ratio2(cgIdx,:)
    real_data2 = imputer.fit_transform(real_data2)  # replace NaNs
    # time vector
    time_in_seconds2 = arr2.get("hasPointsTime", np.arange(real_data2.shape[0]))
    time_in_seconds2 = time_in_seconds2.reshape(-1, 1)
    time_in_seconds2 = torch.tensor(time_in_seconds2).to(torch.float32)
    num_named2 = 0
    worm2_ID = {i: str(i) for i in range(real_data2.shape[1])}
    worm2_ID = dict((v, k) for k, v in worm2_ID.items())
    # normalize the data
    sc = transform
    real_data2 = sc.fit_transform(real_data2)
    real_data2 = torch.tensor(real_data2, dtype=torch.float32)
    # resample the data to a fixed time step
    time_in_seconds2, real_data2 = interpolate_data(
        time_in_seconds2, real_data2, target_dt=resample_dt
    )
    # calculate the time step
    dt2 = torch.zeros_like(time_in_seconds2)
    dt2[1:] = time_in_seconds2[1:] - time_in_seconds2[:-1]
    # recalculate max_timesteps and num_neurons
    max_time2, num_neurons2 = real_data2.shape
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time2, num_neurons2, num_named2),
        end="\n\n",
    )
    # smooth the data
    smooth_real_data0, residual0, smooth_residual0 = smooth_data_preprocess(
        real_data0,
        time_in_seconds0,
        smooth_method,
        dt=np.meadian(dt0),
    )
    smooth_real_data1, residual1, smooth_residual1 = smooth_data_preprocess(
        real_data1,
        time_in_seconds1,
        smooth_method,
        dt=np.median(dt1),
    )
    smooth_real_data2, residual2, smooth_residual2 = smooth_data_preprocess(
        real_data2,
        time_in_seconds2,
        smooth_method,
        dt=np.median(dt2),
    )
    # pickle the data
    data_dict = {
        "worm0": {
            "dataset": "Nguyen2017",
            "smooth_method": smooth_method.upper(),
            "worm": "worm0",
            "calcium_data": real_data0,
            "smooth_calcium_data": smooth_real_data0,
            "residual_calcium": residual0,
            "smooth_residual_calcium": smooth_residual0,
            "neuron_to_idx": worm0_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm0_ID.items()),
            "max_timesteps": max_time0,
            "time_in_seconds": time_in_seconds0,
            "dt": dt0,
            "num_neurons": num_neurons0,
            "num_named_neurons": num_named0,
            "num_unknown_neurons": num_neurons0 - num_named0,
        },
        "worm1": {
            "dataset": "Nguyen2017",
            "smooth_method": smooth_method.upper(),
            "worm": "worm1",
            "calcium_data": real_data1,
            "smooth_calcium_data": smooth_real_data1,
            "residual_calcium": residual1,
            "smooth_residual_calcium": smooth_residual1,
            "neuron_to_idx": worm1_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm1_ID.items()),
            "max_timesteps": max_time1,
            "time_in_seconds": time_in_seconds1,
            "dt": dt1,
            "num_neurons": num_neurons1,
            "num_named_neurons": num_named1,
            "num_unknown_neurons": num_neurons1 - num_named1,
        },
        "worm2": {
            "dataset": "Nguyen2017",
            "smooth_method": smooth_method.upper(),
            "worm": "worm2",
            "calcium_data": real_data2,
            "smooth_calcium_data": smooth_real_data2,
            "residual_calcium": residual2,
            "smooth_residual_calcium": smooth_residual2,
            "neuron_to_idx": worm2_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm2_ID.items()),
            "max_timesteps": max_time2,
            "time_in_seconds": time_in_seconds2,
            "dt": dt2,
            "num_neurons": num_neurons2,
            "num_named_neurons": num_named2,
            "num_unknown_neurons": num_neurons2 - num_named2,
        },
    }
    for worm in data_dict.keys():
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    file = os.path.join(processed_path, "Nguyen2017.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nguyen2017 = pickle.load(pickle_in)
    print(Nguyen2017.keys(), end="\n\n")


def pickle_Skora2018(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Skora et al., Cell Reports 2018,
    Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans.
    """
    data_dict = dict()

    # 'WT_fasted'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Skora2018", "WT_fasted.mat"))[
        "WT_fasted"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        i_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[i]]
        i_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(i_IDs)
        ]
        _, inds = np.unique(
            i_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(i_IDs)
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
        time_in_seconds = timeVectorSeconds[i].reshape(timeVectorSeconds[i].shape[0], 1)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Skora2018",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'WT_starved'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Skora2018", "WT_starved.mat"))[
        "WT_starved"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for ii, real_data in enumerate(all_traces):
        worm = "worm" + str(ii + i + 1)
        ii_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[ii]]
        ii_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(ii_IDs)
        ]
        _, inds = np.unique(
            ii_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        ii_IDs = [ii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(ii_IDs)
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
        time_in_seconds = timeVectorSeconds[ii].reshape(
            timeVectorSeconds[ii].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Skora2018",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Skora2018.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Skora2018 = pickle.load(pickle_in)
    print(Skora2018.keys(), end="\n\n")


def pickle_Kaplan2020(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Kaplan et al., Neuron 2020,
    Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales.
    """
    data_dict = dict()

    # 'RIShisCl_Neuron2019'
    # load the first .mat file
    arr = mat73.loadmat(
        os.path.join(source_path, "Kaplan2020", "Neuron2019_Data_RIShisCl.mat")
    )["RIShisCl_Neuron2019"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["neuron_ID"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr[
        "traces_bleach_corrected"
    ]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["time_vector"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        _, inds = np.unique(
            all_IDs[i], return_index=True
        )  # only keep indices of unique neuron IDs
        all_IDs[i] = [all_IDs[i][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {nid: str(j) for nid, j in enumerate(all_IDs[i])}
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        time_in_seconds = timeVectorSeconds[i].reshape(timeVectorSeconds[i].shape[0], 1)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'MNhisCl_RIShisCl_Neuron2019'
    # load the second .mat file
    arr = mat73.loadmat(
        os.path.join(source_path, "Kaplan2020", "Neuron2019_Data_MNhisCl_RIShisCl.mat")
    )["MNhisCl_RIShisCl_Neuron2019"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["neuron_ID"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr[
        "traces_bleach_corrected"
    ]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["time_vector"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for ii, real_data in enumerate(all_traces):
        worm = "worm" + str(ii + i + 1)
        _, inds = np.unique(
            all_IDs[ii], return_index=True
        )  # only keep indices of unique neuron IDs
        all_IDs[ii] = [all_IDs[ii][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {nid: str(j) for nid, j in enumerate(all_IDs[ii])}
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        time_in_seconds = timeVectorSeconds[ii].reshape(
            timeVectorSeconds[ii].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'MNhisCl_RIShisCl_Neuron2019'
    # load the third .mat file
    arr = mat73.loadmat(
        os.path.join(source_path, "Kaplan2020", "Neuron2019_Data_SMDhisCl_RIShisCl.mat")
    )["SMDhisCl_RIShisCl_Neuron2019"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["neuron_ID"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr[
        "traces_bleach_corrected"
    ]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["time_vector"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for iii, real_data in enumerate(all_traces):
        worm = "worm" + str(iii + ii + 1 + i + 1)
        _, inds = np.unique(
            all_IDs[iii], return_index=True
        )  # only keep indices of unique neuron IDs
        all_IDs[iii] = [all_IDs[iii][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {nid: str(j) for nid, j in enumerate(all_IDs[iii])}
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        time_in_seconds = timeVectorSeconds[iii].reshape(
            timeVectorSeconds[iii].shape[0], 1
        )
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Kaplan2020.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kaplan2020 = pickle.load(pickle_in)
    print(Kaplan2020.keys(), end="\n\n")


def pickle_Uzel2022(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Uzel et al 2022., Cell CurrBio 2022,
    A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans.
    """
    data_dict = dict()
    # load .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Uzel2022", "Uzel_WT.mat"))[
        "Uzel_WT"
    ]  # load .mat file
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["tv"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        i_IDs = [np.array(j).item() for j in all_IDs[i]]
        _, inds = np.unique(
            i_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(int(j)) if type(j) != str else j) for nid, j in enumerate(i_IDs)
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
        time_in_seconds = timeVectorSeconds[i].reshape(timeVectorSeconds[i].shape[0], 1)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Uzel2022",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to time x 302
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Uzel2022.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Uzel2022 = pickle.load(pickle_in)
    print(Uzel2022.keys(), end="\n\n")


def pickle_Flavell2023(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Flavell et al., bioRxiv 2023,
    Brain-wide representations of behavior spanning multiple timescales and states in C. elegans.
    """
    # imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    data_dict = dict()
    data_dir = os.path.join(source_path, "Flavell2023")
    # process all .h5 files in the data directory
    for i, h5_file in enumerate(os.listdir(data_dir)):
        if not h5_file.endswith(".h5"):
            continue
        # each h5 has the data for one (1) worm
        h5_file = os.path.join(data_dir, h5_file)
        worm = "worm" + str(i)
        h5 = h5py.File(h5_file, "r")
        time_in_seconds = torch.tensor(h5["timestamp_confocal"]).to(torch.float32)
        time_in_seconds = time_in_seconds - time_in_seconds[0]  # start at 0
        time_in_seconds = time_in_seconds.reshape((-1, 1))
        if i == 0:
            print(list(h5.keys()), end="\n\n")
        print("num. worms:", 1, end="\n\n")
        # get calcium data for this worm
        calcium_data = np.array(
            h5["trace_array_F20"], dtype=float
        )  # GCaMP neural activity traced normalized by 20th percentile
        # get neuron labels
        neurons = np.array(
            h5["neuropal_label"], dtype=str
        )  # list of full labels (if neuron wasn't labeled the entry is "missing")
        # flip a coin to chose L/R for unsure bilaterally symmetric neurons
        neurons_copy = []
        for neuron in neurons:
            if neuron.replace("?", "L") not in set(neurons_copy):
                neurons_copy.append(neuron.replace("?", "L"))
            else:
                neurons_copy.append(neuron.replace("?", "R"))
        neurons = np.array(neurons_copy)
        # extract neurons with labels
        named_inds = np.where(neurons != "missing")[0]
        num_named = len(named_inds)
        neuron_to_idx = {
            (neuron if idx in named_inds else str(idx)): idx
            for idx, neuron in enumerate(neurons)
        }
        # normalize the data
        sc = transform
        calcium_data = sc.fit_transform(calcium_data)
        calcium_data = torch.tensor(calcium_data, dtype=torch.float32)
        # resample the data to a fixed time step
        time_in_seconds, calcium_data = interpolate_data(
            time_in_seconds, calcium_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = calcium_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            calcium_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        # add worm to data dictionary
        data_dict.update(
            {
                worm: {
                    "dataset": "Flavell2023",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": calcium_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": max_timesteps,
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                }
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Flavell2023.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Flavell2023 = pickle.load(pickle_in)
    print(Flavell2023.keys(), end="\n\n")
    return data_dict


def pickle_Leifer2023(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Randi, ..., Leifer et al.,
    bioRxiv 2023, Neural signal propagation atlas of C. elegans.
    """
    data_dict = dict()
    data_dir = os.path.join(source_path, "Leifer2023")
    files = os.listdir(data_dir)
    num_worms = int(len(files) / 6)  # every worm has 6 txt files

    for i in range(0, num_worms):
        # worm27 doesn't have neuron labels
        if i == 27:
            continue

        if i < 27:
            worm = "worm" + str(i)
        else:
            worm = "worm" + str(i - 1)

        real_data = []
        with open(os.path.join(data_dir, str(i) + "_gcamp.txt"), "r") as f:
            for line in f.readlines():
                cal = list(map(float, line.split(" ")))
                real_data.append(cal)
        real_data = np.array(real_data)  # format: (time, neuron)
        # skip worms with very short recordings
        if real_data.shape[0] < 1000:
            continue

        label_list = []
        with open(os.path.join(data_dir, str(i) + "_labels.txt"), "r") as f:
            for line in f.readlines():
                l = line.strip("\n")
                label_list.append(l)

        # get numbers of neurons and initialize mapping
        num_unnamed = 0
        num_named = real_data.shape[1] - num_unnamed
        label_list = label_list[: real_data.shape[1]]
        neuron_to_idx = dict()

        # compute the time vectoy
        timeVectorSeconds = []
        with open(os.path.join(data_dir, str(i) + "_t.txt"), "r") as f:
            for line in f.readlines():
                l = line.strip("\n")
                timeVectorSeconds.append(str_to_float(l))
        time_in_seconds = np.array(timeVectorSeconds)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32).unsqueeze(1)

        # iterat through labelled neurons
        for j, item in enumerate(label_list):
            previous_list = label_list[:j]
            # if the neuron is unnamed, give it a number larger than 302
            if item == "" or item == "smthng else":
                label_list[j] = str(j + 302)
                num_unnamed += 1
                neuron_to_idx[str(j + 302)] = j
            else:
                # if the neuron is named, and the name is unique, add it to the dictionary
                if item in NEURONS_302 and item not in previous_list:
                    neuron_to_idx[item] = j
                # if the neuron is named, but the name is not unique, give it a number larger than 302
                elif item in NEURONS_302 and item in previous_list:
                    label_list[j] = str(j + 302)
                    num_unnamed += 1
                    neuron_to_idx[str(j + 302)] = j
                else:
                    # if the neuron is recorded without L or R, choose one valid name for it
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
                        label_list[j] = str(j + 302)
                        num_unnamed += 1
                        neuron_to_idx[str(j + 302)] = j

        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # replace nan and inf with 0
        real_data = torch.nan_to_num(real_data, nan=0.0, posinf=0.0, neginf=0.0)

        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )

        data_dict.update(
            {
                worm: {
                    "dataset": "Leifer2023",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_unnamed,
                },
            }
        )

        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
        data_dict[worm]["num_named_neurons"] = (
            data_dict[worm]["named_neurons_mask"].sum().item()
        )
        data_dict[worm]["num_unknown_neurons"] = (
            data_dict[worm]["num_neurons"] - data_dict[worm]["num_named_neurons"]
        )

    # pickle the data
    file = os.path.join(processed_path, "Leifer2023.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Leifer2023 = pickle.load(pickle_in)
    print(Leifer2023.keys(), end="\n\n")
    return data_dict
