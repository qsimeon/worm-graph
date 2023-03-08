from preprocess._pkg import *


class DiffTVR:
    def __init__(self, n: int, dx: float):
        """Differentiate with TVR.

        Args:
            n (int): Number of points in data.
            dx (float): Spacing of data.
        """
        self.n = n
        self.dx = dx

        self.d_mat = self._make_d_mat()
        self.a_mat = self._make_a_mat()
        self.a_mat_t = self._make_a_mat_t()

    def _make_d_mat(self) -> np.array:
        """Make differentiation matrix with central differences. NOTE: not efficient!

        Returns:
            np.array: N x N+1
        """

        arr = np.zeros((self.n, self.n + 1))
        for i in range(0, self.n):
            arr[i, i] = -1.0
            arr[i, i + 1] = 1.0
        return arr / self.dx

    # TODO: improve these matrix constructors
    def _make_a_mat(self) -> np.array:
        """Make integration matrix with trapezoidal rule. NOTE: not efficient!

        Returns:
            np.array: N x N+1
        """
        arr = np.zeros((self.n + 1, self.n + 1))
        for i in range(0, self.n + 1):
            if i == 0:
                continue
            for j in range(0, self.n + 1):
                if j == 0:
                    arr[i, j] = 0.5
                elif j < i:
                    arr[i, j] = 1.0
                elif i == j:
                    arr[i, j] = 0.5

        return arr[1:] * self.dx

    def _make_a_mat_t(self) -> np.array:
        """Transpose of the integration matirx with trapezoidal rule. NOTE: not efficient!

        Returns:
            np.array: N+1 x N
        """
        smat = np.ones((self.n + 1, self.n))

        cmat = np.zeros((self.n, self.n))
        li = np.tril_indices(self.n)
        cmat[li] = 1.0

        dmat = np.diag(np.full(self.n, 0.5))

        vec = np.array([np.full(self.n, 0.5)])
        combmat = np.concatenate((vec, cmat - dmat))

        return (smat - combmat) * self.dx

    def make_en_mat(self, deriv_curr: np.array) -> np.array:
        """Diffusion matrix

        Args:
            deriv_curr (np.array): Current derivative of length N+1

        Returns:
            np.array: N x N
        """
        eps = pow(10, -6)
        vec = 1.0 / np.sqrt(pow(self.d_mat @ deriv_curr, 2) + eps)
        return np.diag(vec)

    def make_ln_mat(self, en_mat: np.array) -> np.array:
        """Diffusivity term

        Args:
            en_mat (np.array): Result from make_en_mat

        Returns:
            np.array: N+1 x N+1
        """
        return self.dx * np.transpose(self.d_mat) @ en_mat @ self.d_mat

    def make_gn_vec(
            self, deriv_curr: np.array, data: np.array, alpha: float, ln_mat: np.array
    ) -> np.array:
        """Negative right hand side of linear problem

        Args:
            deriv_curr (np.array): Current derivative of size N+1
            data (np.array): Data of size N
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat

        Returns:
            np.array: Vector of length N+1
        """
        return (
                self.a_mat_t @ self.a_mat @ deriv_curr
                - self.a_mat_t @ (data - data[0])
                + alpha * ln_mat @ deriv_curr
        )

    def make_hn_mat(self, alpha: float, ln_mat: np.array) -> np.array:
        """Matrix in linear problem

        Args:
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat

        Returns:
            np.array: N+1 x N+1
        """
        return self.a_mat_t @ self.a_mat + alpha * ln_mat

    def get_deriv_tvr_update(
            self, data: np.array, deriv_curr: np.array, alpha: float
    ) -> np.array:
        """Get the TVR update

        Args:
            data (np.array): Data of size N
            deriv_curr (np.array): Current deriv of size N+1
            alpha (float): Regularization parameter

        Returns:
            np.array: Update vector of size N+1
        """

        n = len(data)

        en_mat = self.make_en_mat(deriv_curr=deriv_curr)

        ln_mat = self.make_ln_mat(en_mat=en_mat)

        hn_mat = self.make_hn_mat(alpha=alpha, ln_mat=ln_mat)

        gn_vec = self.make_gn_vec(
            deriv_curr=deriv_curr, data=data, alpha=alpha, ln_mat=ln_mat
        )

        return solve(hn_mat, -gn_vec)

    def get_deriv_tvr(
            self,
            data: np.array,
            deriv_guess: np.array,
            alpha: float,
            no_opt_steps: int,
            return_progress: bool = False,
            return_interval: int = 1,
    ) -> Tuple[np.array, np.array]:
        """Get derivative via TVR over optimization steps

        Args:
            data (np.array): Data of size N
            deriv_guess (np.array): Guess for derivative of size N+1
            alpha (float): Regularization parameter
            no_opt_steps (int): No. opt steps to run
            return_progress (bool, optional): True to return derivative progress during optimization. Defaults to False.
            return_interval (int, optional): Interval at which to store derivative if returning. Defaults to 1.

        Returns:
            Tuple[np.array,np.array]: First is the final derivative of size N+1, second is the stored derivatives if return_progress=True of size no_opt_steps+1 x N+1, else [].
        """

        deriv_curr = deriv_guess

        if return_progress:
            deriv_st = np.full((no_opt_steps + 1, len(deriv_guess)), 0)
        else:
            deriv_st = np.array([])

        for opt_step in range(0, no_opt_steps):
            update = self.get_deriv_tvr_update(
                data=data, deriv_curr=deriv_curr, alpha=alpha
            )

            deriv_curr += update

            if return_progress:
                if opt_step % return_interval == 0:
                    deriv_st[int(opt_step / return_interval)] = deriv_curr

        return (deriv_curr, deriv_st)


def smooth_data_preprocess(calcium_data, smooth_method, dt=1.0):
    """
    Smooth the data, get signals denoised

    Args:
        calcium_data: original calcium data from dataset
        smooth_method: the way to smooth data
        dt: (required when use FFT as smooth_method) the sampling time (unit: sec)

    Returns:
        smooth_ca_data: calcium data that are smoothed
        residual: original residual (calculated by calcium_data)
        residual_smooth_ca_data: residual calculated by smoothed calcium data
    """
    n = calcium_data.shape[0]
    # initialize the size for smooth_calcium_data
    smooth_ca_data = torch.zeros_like(calcium_data)
    # calculate original residual
    residual = calcium_data[1:] - calcium_data[:n - 1]
    if str(smooth_method).lower() == "sg" or smooth_method == None:
        smooth_ca_data = savgol_filter(calcium_data, 5, 3, mode="nearest", axis=-1)
    elif str(smooth_method).lower() == "fft":
        data_torch = calcium_data
        smooth_ca_data = torch.zeros_like(calcium_data)
        max_time, num_neurons = data_torch.shape
        frequencies = torch.fft.rfftfreq(max_time, d=dt)  # dt: sampling time
        threshold = torch.abs(frequencies)[int(frequencies.shape[0] * 0.1)]
        oneD_kernel = torch.abs(frequencies) < threshold
        fft_input = torch.fft.rfftn(data_torch, dim=0)
        oneD_kernel = oneD_kernel.repeat(calcium_data.shape[1], 1).T
        fft_result = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
        smooth_ca_data[0:min(fft_result.shape[0], calcium_data.shape[0])] = fft_result
    elif str(smooth_method).lower() == "tvr":
        diff_tvr = DiffTVR(n, 1)
        for i in range(0, calcium_data.shape[1]):
            temp = np.array(calcium_data[:, i])
            temp.reshape(len(temp), 1)
            (item_denoise, _) = diff_tvr.get_deriv_tvr(
                data=temp,
                deriv_guess=np.full(n + 1, 0.0),
                alpha=0.005,
                no_opt_steps=100,
            )
            smooth_ca_data[:, i] = torch.tensor(item_denoise[: (len(item_denoise) - 1)])
    else:
        print("Wrong Input, check the config/preprocess.yml")
        exit(0)
    m = smooth_ca_data.shape[0]
    residual_smooth_ca_data = smooth_ca_data[1:] - smooth_ca_data[:m - 1]
    return smooth_ca_data, residual, residual_smooth_ca_data


def preprocess_connectome(raw_dir, raw_files):
    """
    If the `graph_tensors.pt` file is not found, this function gets
    called to create it from the raw open source connectome data.
    The connectome data useed here is from Cook et al., 2019 downloaded from
    https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip.
    The data in .mat files was processed into .csv files containing the chemical and
    electrical connectivity for the adult hermaphrodite C. elegans.
    """
    # checking
    assert all([os.path.exists(os.path.join(raw_dir, rf)) for rf in raw_files])
    # list of names of all C. elegans neurons
    neurons_all = set(NEURONS_302)
    # chemical synapses
    GHermChem_Edges = pd.read_csv(os.path.join(raw_dir, "GHermChem_Edges.csv"))  # edges
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    # gap junctions
    GHermElec_Sym_Edges = pd.read_csv(
        os.path.join(raw_dir, "GHermElec_Sym_Edges.csv")
    )  # edges
    GHermElec_Sym_Nodes = pd.read_csv(
        os.path.join(raw_dir, "GHermElec_Sym_Nodes.csv")
    )  # nodes
    # neurons involved in gap junctions
    df = GHermElec_Sym_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Ggap_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()
    # neurons involved in chemical synapses
    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()
    # gap junctions
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
    # chemical synapses
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
    # map neuron names (IDs) to indices
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
    # generate random data TODO: inject real data istead !
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
    # save the mapping of encodings to type of neuron
    codes = np.unique(y)
    types = np.unique(Gsyn_nodes.Group.values)
    node_type = dict(zip(codes, types))
    # graph for electrical connectivity
    electrical_graph = Data(
        x=x, edge_index=ggap_edge_index, edge_attr=ggap_edge_attr, y=y
    )
    # graph for chemical connectivity
    chemical_graph = Data(
        x=x, edge_index=gsyn_edge_index, edge_attr=gsyn_edge_attr, y=y
    )
    # merge electrical and chemical graphs into a single connectome graph
    edge_index = torch.hstack((electrical_graph.edge_index, chemical_graph.edge_index))
    edge_attr = torch.vstack((electrical_graph.edge_attr, chemical_graph.edge_attr))
    edge_index, edge_attr = coalesce(
        edge_index, edge_attr, reduce="add"
    )  # features = [elec_wt, chem_wt]
    assert all(chemical_graph.y == electrical_graph.y), "Node labels not matched!"
    x = chemical_graph.x
    y = chemical_graph.y
    # basic attributes of PyG Data object
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    # add some additional attributes to the graph
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
    # initialize position dict then replace with atlas coordinates if available
    pos = dict(zip(np.arange(graph.num_nodes), np.zeros(shape=(graph.num_nodes, 2))))
    for k, v in zip(keys, values):
        pos[k] = v
    # assign each node its global node index
    n_id = torch.arange(graph.num_nodes)
    # save the tensors to use as raw data in the future.
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


def reshape_calcium_data(single_worm_dataset):
    """
    Modifies the worm dataset to restructure calcium data
    into a standard shape of max_time x 302. Inserts neuron
    masks and mappings of neuron labels to indices in the data.
    """
    # get the calcium data for this worm
    origin_calcium_data = single_worm_dataset["calcium_data"]
    smooth_calcium_data = single_worm_dataset["smooth_calcium_data"]
    residual_calcium = single_worm_dataset["residual_calcium"]
    residual_smooth_calcium = single_worm_dataset["residual_smooth_calcium"]
    # get the number of unidentified tracked neurons
    num_unknown_neurons = single_worm_dataset["num_unknown_neurons"]
    # get the neuron to idx map
    neuron_to_idx = single_worm_dataset["neuron_to_idx"]
    idx_to_neuron = single_worm_dataset["idx_to_neuron"]
    # get the length of the time series
    max_time = single_worm_dataset["max_time"]
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
    standard_calcium_data = torch.zeros(max_time, 302, dtype=origin_calcium_data.dtype)
    standard_residual_calcium = torch.zeros(max_time-1, 302, dtype=residual_calcium.dtype)
    standard_smooth_calcium_data = torch.zeros(max_time, 302, dtype=smooth_calcium_data.dtype)
    standard_residual_smooth_calcium = torch.zeros(max_time-1, 302, dtype=residual_smooth_calcium.dtype)
    # fill the new calcium data structure with data from named neurons
    slot_to_named_neuron = dict((k, v) for k, v in enumerate(neurons_302))
    for slot, neuron in slot_to_named_neuron.items():
        if neuron in neuron_to_idx:  # named neuron
            idx = neuron_to_idx[neuron]
            named_neuron_to_idx[neuron] = idx
            standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
            standard_residual_calcium[:, slot] = residual_calcium[:, idx]
            standard_smooth_calcium_data[:, slot] = smooth_calcium_data[:, idx]
            standard_residual_smooth_calcium[:, slot] = residual_smooth_calcium[:, idx]
            named_neurons_mask[slot] = True
    # randomly distribute the remaining data from unknown neurons
    for neuron in set(neuron_to_idx) - set(named_neuron_to_idx):
        unknown_neuron_to_idx[neuron] = neuron_to_idx[neuron]
    free_slots = list(np.where(~named_neurons_mask)[0])
    slot_to_unknown_neuron = dict(
        zip(
            np.random.choice(free_slots, num_unknown_neurons, replace="False"),
            unknown_neuron_to_idx.keys(),
        )
    )
    for slot, neuron in slot_to_unknown_neuron.items():
        idx = unknown_neuron_to_idx[neuron]
        standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
        standard_residual_calcium[:, slot] = residual_calcium[:, idx]
        standard_smooth_calcium_data[:, slot] = smooth_calcium_data[:, idx]
        standard_residual_smooth_calcium[:, slot] = residual_smooth_calcium[:, idx]
        unknown_neurons_mask[slot] = True
    # combined slot to neuron mapping
    slot_to_neuron = dict()
    slot_to_neuron.update(slot_to_named_neuron)
    slot_to_neuron.update(slot_to_unknown_neuron)
    # modify the worm dataset to with new attributes
    single_worm_dataset.update(
        {
            "calcium_data": standard_calcium_data,
            "smooth_calcium": standard_smooth_calcium_data,
            "residual_calcium": standard_residual_calcium,
            "residual_smooth_calcium": standard_residual_smooth_calcium,
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
    return single_worm_dataset


def pickle_neural_data(
        url, zipfile, dataset="all", transform=MinMaxScaler(feature_range=(-1, 1)), smooth_method="fft"
):
    """
    Function for converting C. elegans neural data from open source
    datasets into a consistent and compressed form (.pickle) for
    our purposes.
    url: str, a download link to a zip file containing the opensource data in raw form.
    zipfile: str, the name of the zipfile that is being downloaded.
    """
    global source_path, processed_path
    zip_path = os.path.join(ROOT_DIR, zipfile)
    source_path = os.path.join(ROOT_DIR, zipfile.strip(".zip"))
    processed_path = os.path.join(ROOT_DIR, "data/processed/neural")
    # Download the curated open-source worm datasets from host server
    if not os.path.exists(source_path):
        # Downloading can take up to 8 minutes depending on your network speed.
        download_url(url=url, folder=ROOT_DIR, filename=zipfile)
        if dataset.lower() == "all":  # extract all the datasets
            extract_zip(zip_path, folder=source_path)  # extract zip file
        else:
            bash_command = [
                "unzip",
                zip_path,
                "{}/*".format(dataset),
                "-d",
                source_path,
            ]
            std_out = subprocess.run(bash_command, text=True)
            print(std_out, end="\n\n")
        os.unlink(zip_path)  # remove zip file
    # Pickle all the datasets ... OR
    if dataset is None or dataset.lower() == "all":
        for dataset in VALID_DATASETS:
            print("Dataset:", dataset, end="\n\n")
            pickler = eval("pickle_" + dataset)
            pickler(transform)
    # (re)-Pickle a single dataset
    else:
        assert (
                dataset in VALID_DATASETS
        ), "Invalid dataset requested! Please pick one from:\n{}".format(
            list(VALID_DATASETS)
        )
        print("Dataset:", dataset, end="\n\n")
        pickler = eval("pickle_" + dataset)
        pickler(transform, smooth_method)
    # delete the downloaded raw datasets
    shutil.rmtree(source_path)  # files too large to push to GitHub
    # create a file the indicates preprocessing succesful
    open(os.path.join(processed_path, ".processed"), "a").close()
    return None


def pickle_Kato2015(transform, smooth_method="fft"):
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Kato2015",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Kato2015",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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


def pickle_Nichols2017(transform, smooth_method="fft"):
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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


def pickle_Nguyen2017(transform, smooth_method="fft"):
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
    max_time0, num_neurons0 = real_data0.shape
    num_named0 = 0
    worm0_ID = {i: str(i) for i in range(num_neurons0)}
    worm0_ID = dict((v, k) for k, v in worm0_ID.items())
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time0, num_neurons0, num_named0),
        end="\n\n",
    )
    # normalize the data
    sc = transform
    real_data0 = sc.fit_transform(real_data0[:, :num_neurons0])
    real_data0 = torch.tensor(real_data0, dtype=torch.float64)
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
    max_time1, num_neurons1 = real_data1.shape
    num_named1 = 0
    worm1_ID = {i: str(i) for i in range(num_neurons1)}
    worm1_ID = dict((v, k) for k, v in worm1_ID.items())
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time1, num_neurons1, num_named1),
        end="\n\n",
    )
    # normalize the data
    sc = transform
    real_data1 = sc.fit_transform(real_data1[:, :num_neurons1])
    real_data1 = torch.tensor(real_data1, dtype=torch.float64)
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
    max_time2, num_neurons2 = real_data2.shape
    num_named2 = 0
    worm2_ID = {i: str(i) for i in range(num_neurons2)}
    worm2_ID = dict((v, k) for k, v in worm2_ID.items())
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time2, num_neurons2, num_named2),
        end="\n\n",
    )
    # normalize the data
    sc = transform
    real_data2 = sc.fit_transform(real_data2[:, :num_neurons2])
    real_data2 = torch.tensor(real_data2, dtype=torch.float64)

    smooth_real_data0, residual0, smooth_residual0 = smooth_data_preprocess(real_data0, smooth_method)
    smooth_real_data1, residual1, smooth_residual1 = smooth_data_preprocess(real_data1, smooth_method)
    smooth_real_data2, residual2, smooth_residual2 = smooth_data_preprocess(real_data2, smooth_method)
    # pickle the data
    data_dict = {
        "worm0": {
            "dataset": "Nguyen2017",
            "worm": "worm0",
            "calcium_data": real_data0,
            "smooth_calcium_data": smooth_real_data0,
            "residual_calcium": residual0,
            "residual_smooth_calcium": smooth_residual0,
            "neuron_to_idx": worm0_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm0_ID.items()),
            "max_time": max_time0,
            "num_neurons": num_neurons0,
            "num_named_neurons": num_named0,
            "num_unknown_neurons": num_neurons0 - num_named0,
        },
        "worm1": {
            "dataset": "Nguyen2017",
            "worm": "worm1",
            "calcium_data": real_data1,
            "smooth_calcium_data": smooth_real_data1,
            "residual_calcium": residual1,
            "residual_smooth_calcium": smooth_residual1,
            "neuron_to_idx": worm1_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm1_ID.items()),
            "max_time": max_time1,
            "num_neurons": num_neurons1,
            "num_named_neurons": num_named1,
            "num_unknown_neurons": num_neurons1 - num_named1,
        },
        "worm2": {
            "dataset": "Nguyen2017",
            "worm": "worm2",
            "calcium_data": real_data2,
            "smooth_calcium_data": smooth_real_data2,
            "residual_calcium": residual2,
            "residual_smooth_calcium": smooth_residual2,
            "neuron_to_idx": worm2_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm2_ID.items()),
            "max_time": max_time2,
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


def pickle_Skora2018(transform, smooth_method="fft"):
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Skora2018",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Skora2018",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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


def pickle_Kaplan2020(transform, smooth_method="fft"):
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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


def pickle_Uzel2022(transform, smooth_method="fft"):
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
        max_time, num_neurons = real_data.shape
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data[:, :num_neurons])
        real_data = torch.tensor(
            real_data, dtype=torch.float64
        )  # add a feature dimension and convert to tensor

        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(real_data, smooth_method)

        data_dict.update(
            {
                worm: {
                    "dataset": "Uzel2022",
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": int(max_time),
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


def pickle_Flavell2023(transform, smooth_method="fft"):
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
        calcium_data = torch.tensor(calcium_data, dtype=torch.float64)
        max_time, num_neurons = calcium_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_time, num_neurons, num_named),
            end="\n\n",
        )
        # add worm to data dictionary
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(calcium_data, smooth_method)
        data_dict.update(
            {
                worm: {
                    "dataset": "Flavell2023",
                    "worm": "worm0",
                    "calcium_data": calcium_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "residual_smooth_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
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
