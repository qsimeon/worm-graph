from preprocess._pkg import *


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
    into a standard shape of 302 x max_time. Inserts neuron
    masks and mappings of neuron labels to indices in the data.
    """
    # get the calcium data for this worm
    origin_calcium_data = single_worm_dataset["calcium_data"]
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
    standard_calcium_data = torch.zeros(max_time, 302, dtype=origin_calcium_data.dtype)
    # fill the new calcium data structure with data from named neurons
    slot_to_named_neuron = dict((k, v) for k, v in enumerate(neurons_302))
    for slot, neuron in slot_to_named_neuron.items():
        if neuron in neuron_to_idx:  # named neuron
            idx = neuron_to_idx[neuron]
            named_neuron_to_idx[neuron] = idx
            standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
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
        unknown_neurons_mask[slot] = True
    # combined slot to neuron mapping
    slot_to_neuron = dict()
    slot_to_neuron.update(slot_to_named_neuron)
    slot_to_neuron.update(slot_to_unknown_neuron)
    # modify the worm dataset to with new attributes
    single_worm_dataset.update(
        {
            "calcium_data": standard_calcium_data,
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
    url, zipfile, dataset="all", transform=MinMaxScaler(feature_range=(-1, 1))
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
        pickler(transform)
    # delete the downloaded raw datasets
    shutil.rmtree(source_path)  # files too large to push to GitHub
    # create a file the indicates preprocessing succesful
    open(os.path.join(processed_path, ".processed"), "a").close()
    return None


def pickle_Kato2015(transform):
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
        data_dict.update(
            {
                worm: {
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Kato2015",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Kato2015.pickle")
    pickle_out = open(file, "wb")
    dataset_object = {
        "name": "Kato2015",
        "num_worms": len(data_dict),
        "generator": iter(data_dict.items()),
    }
    pickle.dump(dataset_object, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kato2015 = pickle.load(pickle_in)
    print(Kato2015.items(), end="\n\n")


def pickle_Nichols2017(transform):
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Nichols2017",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Nichols2017.pickle")
    pickle_out = open(file, "wb")
    dataset_object = {
        "name": "Nichols2017",
        "num_worms": len(data_dict),
        "generator": iter(data_dict.items()),
    }
    pickle.dump(dataset_object, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nichols2017 = pickle.load(pickle_in)
    print(Nichols2017.items(), end="\n\n")


def pickle_Nguyen2017(transform):
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
    # get data for worm 1
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
    # pickle the data
    data_dict = {
        "worm0": {
            "dataset": "Nguyen2017",
            "worm": "worm0",
            "calcium_data": real_data0,
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
    dataset_object = {
        "name": "Nguyen2017",
        "num_worms": len(data_dict),
        "generator": iter(data_dict.items()),
    }
    pickle.dump(dataset_object, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nguyen2017 = pickle.load(pickle_in)
    print(Nguyen2017.items(), end="\n\n")


def pickle_Skora2018(transform):
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Skora2018",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Skora2018",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Skora2018.pickle")
    pickle_out = open(file, "wb")
    dataset_object = {
        "name": "Skora2018",
        "num_worms": len(data_dict),
        "generator": iter(data_dict.items()),
    }
    pickle.dump(dataset_object, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Skora2018 = pickle.load(pickle_in)
    print(Skora2018.items(), end="\n\n")


def pickle_Kaplan2020(transform):
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Kaplan2020",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Kaplan2020.pickle")
    pickle_out = open(file, "wb")
    dataset_object = {
        "name": "Kaplan2020",
        "num_worms": len(data_dict),
        "generator": iter(data_dict.items()),
    }
    pickle.dump(dataset_object, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kaplan2020 = pickle.load(pickle_in)
    print(Kaplan2020.items(), end="\n\n")


def pickle_Uzel2022(transform):
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
        data_dict.update(
            {
                worm: {
                    "dataset": "Uzel2022",
                    "worm": worm,
                    "calcium_data": real_data,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_time": max_time,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                },
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Uzel2022.pickle")
    pickle_out = open(file, "wb")
    dataset_object = {
        "name": "Uzel2022",
        "num_worms": len(data_dict),
        "generator": iter(data_dict.items()),
    }
    pickle.dump(dataset_object, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Uzel2022 = pickle.load(pickle_in)
    print(Uzel2022.items(), end="\n\n")
