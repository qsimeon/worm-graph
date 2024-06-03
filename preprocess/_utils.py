from preprocess._pkg import *

# Initialize logger
logger = logging.getLogger(__name__)


### Function definitions # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def pickle_neural_data(
    url,
    zipfile,
    source_dataset="all",
    # TODO: Try different transforms from sklearn such as QuantileTransformer, etc. as well as custom CausalNormalizer.
    transform=StandardScaler(),  # StandardScaler() #PowerTransformer() #CausalNormalizer() #None
    smooth_method="ma",
    interpolate_method="linear",
    resample_dt=None,
    cleanup=False,
    **kwargs,
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
    source_dataset : str, optional (default: 'all')
        The name of the source dataset to be pickled.
        If None or 'all', all datasets are pickled.
    transform : object, optional (default: StandardScaler())
        The sklearn transformation to be applied to the data.
    smooth_method : str, optional (default: 'ma')
        The smoothing method to apply to the data;
        options are 'ga', 'es' or 'ma'.
    interpolate_method: str, optional (default: 'linear')
        The scipy interpolation method to use when resampling the data.
    resample_dt : float, optional (default: None)
        The resampling time interval in seconds.
        If None, no resampling is performed.

    Calls
    -----
    {SourceDataset}Preprocessor : class in preprocess/_utils.py
        The class that preprocesses the data for the specified source dataset.

    """
    zip_path = os.path.join(ROOT_DIR, zipfile)
    source_path = os.path.join(ROOT_DIR, zipfile.strip(".zip"))
    # Make the neural data directory if it doesn't exist
    processed_path = os.path.join(ROOT_DIR, "data/processed/neural")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path, exist_ok=True)
    # If .zip not found in the root directory, download the curated open-source worm datasets
    if not os.path.exists(source_path):
        download_url(url=url, folder=ROOT_DIR, filename=zipfile)
        # Extract all the datasets ... OR
        if source_dataset.lower() == "all":
            # Extract zip file then delete it
            extract_zip(zip_path, folder=source_path, delete_zip=True)
        # Extract just the requested source dataset
        else:
            bash_command = [
                "unzip",
                zip_path,
                "{}/*".format(source_dataset),
                "-d",
                source_path,
                "-x",
                "__MACOSX/*",
            ]
            # Run the bash command
            std_out = subprocess.run(bash_command, text=True)
            # Output to log or terminal
            logger.info(std_out)
            # Delete the zip file
            os.unlink(zip_path)
    # (re)-Pickle all the datasets ... OR
    if source_dataset is None or source_dataset.lower() == "all":
        for source in EXPERIMENT_DATASETS:
            logger.info(f"Start processing {source}.")
            try:
                # Instantiate the relevant preprocessor class
                preprocessor = eval(source + "Preprocessor")(
                    transform,
                    smooth_method,
                    interpolate_method,
                    resample_dt,
                    **kwargs,
                )
                # Call its method
                preprocessor.preprocess()
            except NameError:
                continue
        # Create a file to indicate that the preprocessing was successful
        open(os.path.join(processed_path, ".processed"), "a").close()
    # ... (re)-Pickle a single dataset
    else:
        assert (
            source_dataset in EXPERIMENT_DATASETS
        ), "Invalid source dataset requested! Please pick one from:\n{}".format(
            list(EXPERIMENT_DATASETS)
        )
        logger.info(f"Start processing {source_dataset}.")
        try:
            # Instantiate the relevant preprocessor class
            preprocessor = eval(source_dataset + "Preprocessor")(
                transform,
                smooth_method,
                interpolate_method,
                resample_dt,
                **kwargs,
            )
            # Call its method
            preprocessor.preprocess()
        except NameError:
            pass
    # Delete the unzipped folder
    if cleanup:
        shutil.rmtree(source_path)
    return None


### DEBUG ###
def get_presaved_datasets(url, file):
    """
    Download and unzip presaved data splits (commonly requested data patterns).
    Deletes the zip file once the dataset has been extracted to the data folder.
    """
    presaved_url = url
    presaved_file = file
    presave_path = os.path.join(ROOT_DIR, presaved_file)
    data_path = os.path.join(ROOT_DIR, "data")
    download_url(url=presaved_url, folder=ROOT_DIR, filename=presaved_file)
    extract_zip(presave_path, folder=data_path, delete_zip=True)
    return None


### DEBUG ###

def preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr):
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

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
    pos = dict(
        zip(
            np.arange(graph.num_nodes),
            np.zeros(shape=(graph.num_nodes, 2), dtype=np.float32),
        )
    )
    for k, v in zip(keys, values):
        pos[k] = v

    # Assign each node its global node index
    n_id = torch.arange(graph.num_nodes)
    
    return pos, n_id

def preprocess_openworm(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "OpenWormConnectome.csv"))
        
    origin = []
    target = []
    
    edges = []
    edge_attr = []

    # CANL, CANR not considered neurons in Cook et al (still include in matrix - data will be zero/null when passed to model)
    
    for i in range(len(df)):
        neuron1 = df.loc[i, "Origin"]
        neuron2 = df.loc[i, "Target"]
        
        origin += [neuron1]
        target += [neuron2]
        
        type = df.loc[i, "Type"]
        num_connections = df.loc[i, "Number of Connections"]
        
        if [neuron1, neuron2] not in edges:
            edges += [[neuron1, neuron2]]
            if type == "GapJunction":
                edge_attr += [[num_connections, 0]]
            else:
                edge_attr += [[0, num_connections]]
        else:
            if type == "GapJunction":
                edge_attr[-1][0] = num_connections
            else:
                edge_attr[-1][-1] = num_connections
    
    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)
    
    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_witvliet_2020_7(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "witvliet_2020_7_processed.csv"))
    
    origin = []
    target = []
    
    edges = []
    edge_attr = []

    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]
        
        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin += [neuron1]
            target += [neuron2]
            
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]
            
            if [neuron1, neuron2] not in edges:
                edges += [[neuron1, neuron2]]
                if type == "electrical":
                    edge_attr += [[num_connections, 0]]
                else:
                    edge_attr += [[0, num_connections]]
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)
    
    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_witvliet_2020_8(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "witvliet_2020_8_processed.csv"))
    
    origin = []
    target = []
    
    edges = []
    edge_attr = []

    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]
        
        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin += [neuron1]
            target += [neuron2]
            
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]
            
            if [neuron1, neuron2] not in edges:
                edges += [[neuron1, neuron2]]
                if type == "electrical":
                    edge_attr += [[num_connections, 0]]
                else:
                    edge_attr += [[0, num_connections]]
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)
    
    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_cook_2019(raw_dir):
    edges = []
    edge_attr = []

    # chemical synapse processing
    df = pd.read_excel(os.path.join(raw_dir, "Cook2019.xlsx"), sheet_name="hermaphrodite chemical")

    for i, line in enumerate(df):
        if i > 2:
            col_data = df.iloc[:-1, i]
            for j, weight in enumerate(col_data):
                if j > 1 and not pd.isna(df.iloc[j, i]):
                    post = df.iloc[1, i]
                    pre = df.iloc[j, 2]
                    if pre in NEURON_LABELS and post in NEURON_LABELS:
                        edges += [[pre, post]]
                        edge_attr += [[0, df.iloc[j, i]]]

    # gap junction processing
    df = pd.read_excel(os.path.join(raw_dir, "Cook2019.xlsx"), sheet_name="hermaphrodite gap jn asymmetric")

    for i, line in enumerate(df):
        if i > 2:
            col_data = df.iloc[:-1, i]
            for j, weight in enumerate(col_data):
                if j > 1 and not pd.isna(df.iloc[j, i]):
                    post = df.iloc[1, i]
                    pre = df.iloc[j, 2]
                    if pre in NEURON_LABELS and post in NEURON_LABELS:
                        if [pre, post] in edges:
                            edge_idx = edges.index([pre, post])
                            edge_attr[edge_idx][0] = df.iloc[j, i]
                        else:
                            edges += [[pre, post]]
                            edge_attr += [[df.iloc[j, i], 0]]

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)

    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_white_1986_whole(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_whole_processed.csv"))
    origin = []
    target = []
    edges = []

    edge_attr = []

    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]
        
        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin += [neuron1]
            target += [neuron2]
            
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]
            
            if [neuron1, neuron2] not in edges:
                edges += [[neuron1, neuron2]]
                if type == "electrical":
                    edge_attr += [[num_connections, 0]]
                else:
                    edge_attr += [[0, num_connections]]
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)

    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_white_1986_n2u(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_n2u_processed.csv"))
    origin = []
    target = []
    edges = []

    edge_attr = []

    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]
        
        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin += [neuron1]
            target += [neuron2]
            
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]
            
            if [neuron1, neuron2] not in edges:
                edges += [[neuron1, neuron2]]
                if type == "electrical":
                    edge_attr += [[num_connections, 0]]
                else:
                    edge_attr += [[0, num_connections]]
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)
    
    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_white_1986_jsh(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_jsh_processed.csv"))
    origin = []
    target = []
    edges = []

    edge_attr = []

    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]
        
        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin += [neuron1]
            target += [neuron2]
            
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]
            
            if [neuron1, neuron2] not in edges:
                edges += [[neuron1, neuron2]]
                if type == "electrical":
                    edge_attr += [[num_connections, 0]]
                else:
                    edge_attr += [[0, num_connections]]
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)

    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_white_1986_jse(raw_dir):
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_jse_processed.csv"))
    origin = []
    target = []
    edges = []

    edge_attr = []

    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]
        
        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin += [neuron1]
            target += [neuron2]
            
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]
            
            if [neuron1, neuron2] not in edges:
                edges += [[neuron1, neuron2]]
                if type == "electrical":
                    edge_attr += [[num_connections, 0]]
                else:
                    edge_attr += [[0, num_connections]]
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    neuron_to_idx = dict(zip(NEURON_LABELS, [i for i in range(len(NEURON_LABELS))]))
    idx_to_neuron = dict(zip([i for i in range(len(NEURON_LABELS))], NEURON_LABELS))

    edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
    node_type = {0: 'Type1', 1: 'Type2'}
    num_classes = len(node_type)

    # for x, y values
    # Neurons involved in chemical synapses
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes
    neurons_all = set(NEURON_LABELS)

    df = GHermChem_Nodes
    df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
    Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    # num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
    x = torch.randn(len(NEURON_LABELS), 1024, dtype=torch.float)

    edge_attr = torch.tensor(edge_attr)
    pos, n_id = preprocess_common_tasks(raw_dir, x, y, edge_index, edge_attr)

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

def preprocess_default(raw_dir, raw_files):
    # Check if the raw connectome data exists
    if not os.path.exists(raw_dir):
        download_url(url=RAW_DATA_URL, folder=ROOT_DIR, filename=RAW_ZIP)
        extract_zip(
            path=os.path.join(ROOT_DIR, RAW_ZIP),
            folder=RAW_DATA_DIR,
            delete_zip=True,
        )

    # Check that all the necessary raw files were extracted
    assert all([os.path.exists(os.path.join(raw_dir, rf)) for rf in raw_files])
    # Names of all C. elegans hermaphrodite neurons
    # NOTE: Only neurons in this list will be included in the connectomes constructed here.
    neurons_all = set(NEURON_LABELS)
    # Chemical synapses nodes and edges
    GHermChem_Edges = pd.read_csv(os.path.join(raw_dir, "GHermChem_Edges.csv"))  # edges
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"))  # nodes

    # Gap junctions
    GHermElec_Sym_Edges = pd.read_csv(os.path.join(raw_dir, "GHermElec_Sym_Edges.csv"))  # edges
    GHermElec_Sym_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermElec_Sym_Nodes.csv"))  # nodes

    # Neurons involved in gap junctions
    df = GHermElec_Sym_Nodes
    df["Name"] = [
        v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]
    ]  # standard naming
    Ggap_nodes = (
        df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()
    )  # filter out non-neurons
    # Neurons (i.e. nodes) in chemical synapses
    df = GHermChem_Nodes
    df["Name"] = [
        v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]
    ]  # standard naming
    Gsyn_nodes = (
        df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()
    )  # filter out non-neurons
    # Gap junctions edges
    df = GHermElec_Sym_Edges
    df["EndNodes_1"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_1"]]
    df["EndNodes_2"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_2"]]
    inds = [
        i
        for i in GHermElec_Sym_Edges.index
        if df.iloc[i]["EndNodes_1"] in set(Ggap_nodes.Name)
        and df.iloc[i]["EndNodes_2"] in set(Ggap_nodes.Name)
    ]  # indices
    Ggap_edges = df.iloc[inds].reset_index(drop=True)

    # Chemical synapses
    df = GHermChem_Edges
    df["EndNodes_1"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_1"]]
    df["EndNodes_2"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_2"]]
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
        ggap_edge_index[i, :] = torch.tensor([neuron_to_idx[x] for x in row], dtype=torch.long)
    ggap_edge_index = ggap_edge_index.T  # [2, num_edges]

    # edge_index for chemical synapses
    arr = Gsyn_edges[["EndNodes_1", "EndNodes_2"]].values
    gsyn_edge_index = torch.empty(*arr.shape, dtype=torch.long)
    for i, row in enumerate(arr):
        gsyn_edge_index[i, :] = torch.tensor([neuron_to_idx[x] for x in row], dtype=torch.long)
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

    # Generate random data
    # TODO: Inject real data instead!
    x = torch.randn(
        num_nodes, num_node_features, dtype=torch.float
    )  # [num_nodes, num_node_features]

    # data.y target to train against
    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    num_classes = len(le.classes_)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)  # [num_nodes, 1]

    # Save the mapping of encodings to type of neuron
    codes = np.unique(y)
    types = np.unique(Gsyn_nodes.Group.values)
    node_type = dict(zip(codes, types))
    # Normalize outgoing gap junction weights to sum to 1
    ggap_weights = to_dense_adj(edge_index=ggap_edge_index, edge_attr=ggap_edge_attr[:, 0]).squeeze(
        0
    )
    ggap_weights = ggap_weights / torch.clamp(ggap_weights.sum(dim=1, keepdim=True), min=1)
    ggap_edge_index, ggap_edge_attr = dense_to_sparse(ggap_weights)
    ggap_edge_attr = torch.stack((ggap_edge_attr, torch.zeros_like(ggap_edge_attr))).T
    # Normalize outgoing chemical synapse weights to sum to 1
    gsyn_weights = to_dense_adj(edge_index=gsyn_edge_index, edge_attr=gsyn_edge_attr[:, 1]).squeeze(
        0
    )
    gsyn_weights = gsyn_weights / torch.clamp(gsyn_weights.sum(dim=1, keepdim=True), min=1)
    gsyn_edge_index, gsyn_edge_attr = dense_to_sparse(gsyn_weights)
    gsyn_edge_attr = torch.stack((torch.zeros_like(gsyn_edge_attr), gsyn_edge_attr)).T
    # Graph for electrical connectivity uses `torch_geometric.Data` object
    electrical_graph = Data(x=x, edge_index=ggap_edge_index, edge_attr=ggap_edge_attr, y=y)
    # Graph for chemical connectivity uses `torch_geometric.Data` object
    chemical_graph = Data(x=x, edge_index=gsyn_edge_index, edge_attr=gsyn_edge_attr, y=y)
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
    # values = list(df[df.neuron.isin(valids)][["x", "z"]].values)
    values = list(df[df.neuron.isin(valids)][["x", "y", "z"]].values)
    # Initialize position dict then replace with atlas coordinates if available
    pos = dict(
        zip(
            np.arange(graph.num_nodes),
            np.zeros(shape=(graph.num_nodes, 2), dtype=np.float32),
        )
    )
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

# TODO only use pub to determine files
def preprocess_connectome(raw_dir, raw_files, pub=None):
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
    * The connectome data used here is from Cook et al., 2019.
      If the raw data isn't found, please download it at this link:
      https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip
      and drop in the data/raw folder.
    """

    if pub == "openworm":
        preprocess_openworm(raw_dir)
    elif pub == "witvliet_7":
        preprocess_witvliet_2020_7(raw_dir)
    elif pub == "witvliet_8":
        preprocess_witvliet_2020_8(raw_dir)
    elif pub == "white_1986_whole":
        preprocess_white_1986_whole(raw_dir)
    elif pub == "white_1986_n2u":
        preprocess_white_1986_n2u(raw_dir)
    elif pub == "white_1986_jsh":
        preprocess_white_1986_jsh(raw_dir)
    elif pub == "white_1986_jse":
        preprocess_white_1986_jse(raw_dir)
    elif pub == "cook_2019":
        preprocess_cook_2019(raw_dir)
    else:
        # preprocess_default(raw_dir, raw_files)
        preprocess_witvliet_2020_7(raw_dir)
        pass

    return None


def extract_zip(path: str, folder: str = None, log: bool = True, delete_zip: bool = True):
    """
    Extracts a zip archive to a specific folder while ignoring the __MACOSX directory.

    Args:
        path (str): The path to the zip archive.
        folder (str, optional): The folder where the files will be extracted to. Default to the parent of `path`.
        log (bool, optional): If False, will not print anything to the console. Default is True.
        delete_zip (bool, optional): If True, will delete the zip archive after extraction. Default is True.
    """
    if folder is None:
        folder = os.path.dirname(path)
    if log:
        print(f"Extracting {path}...")
    with zipfile.ZipFile(path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if not member.startswith("__MACOSX/"):
                zip_ref.extract(member, folder)
    if delete_zip:
        os.unlink(path)


def gaussian_kernel_smooth(x, t, sigma):
    """Causal Gaussian smoothing for a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        x_smooth (ndarray): The smoothed time series.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    # Apply one-sided exponential decay
    x_smooth = np.zeros_like(x, dtype=np.float32)
    alpha = 1 / (2 * sigma**2)
    # TODO: Vectorize this instead of using a loop.
    for i in range(x.shape[0]):  # temporal dimension
        weights = np.exp(-alpha * np.arange(i, -1, -1) ** 2)
        weights /= weights.sum()
        for j in range(x.shape[1]):  # feature dimension
            x_smooth[i, j] = np.dot(weights, x[: i + 1, j])
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def moving_average_smooth(x, t, window_size):
    """Causal moving average smoothing filter to a multidimensional time series.

    Parameters:
    ----------
        x (ndarray): The input time series to be smoothed.
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        window_size (int): The size of the moving average window. Must be an odd number.

    Returns:
    ----------
        x_smooth (ndarray): The smoothed time series.
    """
    # Ensure window_size is odd for symmetry
    if window_size % 2 == 0:
        window_size += 1
    # Check for correct dimensions
    isnumpy = isinstance(x, np.ndarray)
    if isnumpy:
        x = torch.from_numpy(x)
    dim = x.ndim
    if dim == 1:
        x = x.unsqueeze(-1)
    x_smooth = torch.zeros_like(x)
    # TODO: Vectorize this instead of using a loop.
    for i in range(x.shape[1]):  # feature dimension
        for j in range(x.shape[0]):  # temporal dimension
            start = max(j - window_size // 2, 0)
            end = j + 1
            window = x[start:end, i]
            x_smooth[j, i] = window.mean()
    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if isnumpy:
        x_smooth = x_smooth.cpu().numpy()
    return x_smooth


def exponential_kernel_smooth(x, t, alpha):
    """Exponential kernel smoothing for a multidimensional time series.
    This method method is already causal by its definiton.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        alpha (float): The smoothing factor, 0<alpha<1. A higher value of alpha will
                       result in less smoothing (more weight is given to the current value),
                       while a lower value of alpha will result in more smoothing
                       (more weight is given to the previous smoothed values).

    Returns:
        x_smooth (ndarray): The smoothed time series.
    """
    istensor = isinstance(x, torch.Tensor)
    if istensor:
        x = x.cpu().numpy()
    dim = x.ndim
    if dim == 1:
        x = x.reshape(-1, 1)
    x_smooth = np.zeros_like(x, dtype=np.float32)
    x_smooth[0] = x[0]
    # TODO: Vectorize this smoothing operation
    for i in range(1, x.shape[0]):
        x_smooth[i] = alpha * x[i] + (1 - alpha) * x_smooth[i - 1]

    if dim == 1:
        x_smooth = x_smooth.squeeze(-1)
    if istensor:
        x_smooth = torch.from_numpy(x_smooth)
    return x_smooth


def smooth_data_preprocess(calcium_data, time_in_seconds, smooth_method, **kwargs):
    """Smooths the provided calcium data using the specified smoothing method.
    as a (time, neurons) array `calcium_data`.

    Args:
        calcium_data (np.ndarray): original calcium data with shape (time, neurons)
        time_in_seconds (np.ndarray): time vector with shape (time, 1)
        smooth_method (str): the method used to smooth the data

    Returns:
        smooth_ca_data (np.ndarray): calcium data that is smoothed
    """
    if smooth_method is None:
        smooth_ca_data = calcium_data
    elif str(smooth_method).lower() == "ga":
        smooth_ca_data = gaussian_kernel_smooth(
            calcium_data, time_in_seconds, sigma=kwargs.get("sigma", 5)
        )
    elif str(smooth_method).lower() == "ma":
        smooth_ca_data = moving_average_smooth(
            calcium_data, time_in_seconds, window_size=kwargs.get("window_size", 15)
        )
    elif str(smooth_method).lower() == "es":
        smooth_ca_data = exponential_kernel_smooth(
            calcium_data, time_in_seconds, alpha=kwargs.get("alpha", 0.5)
        )
    else:
        raise TypeError("See `configs/submodule/preprocess.yaml` for viable smooth methods.")
    return smooth_ca_data


def reshape_calcium_data(worm_dataset):
    """Reorganizes calcium data into a standard organized matrix with shape (max_timesteps, NUM_NEURONS).
    Also creates neuron masks and mappings of neuron labels to indices in the data.
    Converts the data to torch tensors.

    Args:
        worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.

    Returns:
        dict: The modified worm dataset with restructured calcium data.
    """
    reshaper = CalciumDataReshaper(worm_dataset)
    return reshaper.worm_dataset


def interpolate_data(time, data, target_dt, method="linear"):
    """Interpolate data using np.interp.

    This function takes the given time points and corresponding data and
    interpolates them to create new data points with the desired time interval.

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
    method : str, optional (default: 'linear')
        The scipy interpolation method to use when resampling the data.

    Returns
    -------
    numpy.ndarray, numpy.ndarray: Two arrays containing the interpolated time points and data.
    """
    # Check if correct interpolation method provided
    assert method in {
        None,
        "linear",
        "quadratic",
        "cubic",
    }, "Invalid interpolation method. Choose from [None, 'linear', 'cubic', 'quadratic']."
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data
    # Ensure that time is a 1D array
    time = time.squeeze()
    # Interpolate the data
    target_time_np = np.arange(time.min(), time.max() + target_dt, target_dt)  # 1D array
    num_neurons = data.shape[1]
    interpolated_data_np = np.zeros((len(target_time_np), num_neurons), dtype=np.float32)
    # TODO: Vectorize this interpolation method
    if method is None:
        target_time_np = time  # 1D array
        interpolated_data_np = data
    elif method == "linear":
        for i in range(num_neurons):
            interpolated_data_np[:, i] = np.interp(target_time_np, time, data[:, i])
    else:  # either quadratic or cubic
        for i in range(num_neurons):
            # NOTE: scipy.interplate.interp1d is deprecated. Best to choose method='linear'.
            interp = interp1d(x=time, y=data[:, i], kind=method)
            interpolated_data_np[:, i] = interp(target_time_np)
    # Reshape interpolated time vector to (time, 1)
    target_time_np = target_time_np.reshape(-1, 1)
    # Return the interpolated data
    return target_time_np, interpolated_data_np


def aggregate_data(time, data, target_dt):
    """Downsample data using aggregation.

    This function takes the given time points and corresponding data and
    downsamples them by averaging over intervals defined by `target_dt`.

    Parameters
    ----------
    time : numpy.ndarray
        1D array containing the time points corresponding to the data,
        with shape (time, 1).
    data : numpy.ndarray
        A 2D array containing the data to be downsampled, with shape
        (time, neurons).
    target_dt : float
        The desired time interval between the downsampled data points.
        If None, no downsampling is performed.

    Returns
    -------
    numpy.ndarray, numpy.ndarray: Two arrays containing the downsampled time points and data.
    """
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data
    # Ensure that time is a 1D array
    time = time.squeeze()
    # Compute the downsample rate
    original_dt = np.median(np.diff(time, axis=0)[1:]).item()
    interval_width = int(target_dt // original_dt)
    # Determine the number of intervals
    num_intervals = len(time) // interval_width
    downsampled_data = np.zeros((num_intervals, data.shape[1]), dtype=np.float32)
    # Create the downsampled time array
    target_time_np = np.arange(time.min(), time.max() + target_dt, target_dt)[:num_intervals]
    # Downsample the data by averaging over intervals
    for i in range(data.shape[1]):
        reshaped_data = data[: num_intervals * interval_width, i].reshape(-1, interval_width)
        downsampled_data[:, i] = reshaped_data.mean(axis=1)
    # Reshape downsampled time vector to (time, 1)
    target_time_np = target_time_np.reshape(-1, 1)
    # Return the interpolated data
    return target_time_np, downsampled_data


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # Class definitions # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class CausalNormalizer:
    """A transform for causal normalization of time series data.

    This normalizer computes the mean and standard deviation up to each time point t,
    ensuring that the normalization at each time point is based solely on past
    and present data, maintaining the causal nature of the time series.
    """

    def __init__(self, nan_fill_method="interpolate"):
        """
        Initialize the CausalNormalizer with a method to handle NaN values.

        Parameters:
        ---
        nan_fill_method (str): Method to fill NaN values. Options are 'ffill' (forward fill),
                                'bfill' (backward fill), and 'interpolate'. Default is 'interpolate'.
        """
        self.cumulative_mean_ = None
        self.cumulative_std_ = None
        self.nan_fill_method = nan_fill_method

    def _handle_nans(self, X):
        """
        Handle NaN values in the dataset X based on the specified method.

        Parameters:
        ----------
        X (array-like): The input data with potential NaN values.

        Returns:
        ----------
        X_filled (array-like): The data with NaN values handled.
        """
        df = pd.DataFrame(X)
        if self.nan_fill_method == "ffill":
            df.fillna(method="ffill", inplace=True)
        elif self.nan_fill_method == "bfill":
            df.fillna(method="bfill", inplace=True)
        elif self.nan_fill_method == "interpolate":
            df.interpolate(method="linear", inplace=True)
        else:
            raise ValueError("Invalid NaN fill method specified.")
        return df.values

    def fit(self, X, y=None):
        """
        Compute the cumulative mean and standard deviation of the dataset X.
        Uses the two-pass algorithm: https://www.wikiwand.com/en/Algorithms_for_calculating_variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._handle_nans(X)
        T, D = X.shape
        cumulative_sum = np.cumsum(X, axis=0)
        cumulative_squares_sum = np.cumsum(X**2, axis=0)
        count = np.arange(1, T + 1).reshape(-1, 1)
        self.cumulative_mean_ = cumulative_sum / count
        cumulative_variance = (
            cumulative_squares_sum
            - 2 * self.cumulative_mean_ * cumulative_sum
            + count * self.cumulative_mean_**2
        ) / (count - 1)
        self.cumulative_std_ = np.sqrt(cumulative_variance)
        # Avoid zero-division
        self.cumulative_std_[self.cumulative_std_ == 0] = 1
        return self

    def transform(self, X):
        """Perform causal normalization on the dataset X using the
        previously computed cumulative mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        if self.cumulative_mean_ is None or self.cumulative_std_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        X_transformed = (X - self.cumulative_mean_) / self.cumulative_std_
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit and transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        return self.fit(X).transform(X)


class CalciumDataReshaper:
    def __init__(self, worm_dataset: dict):
        """
        Notes:
            'idx' refers to the index of the neuron in the original dataset.
                0 < idx < N, where N is however many neurons were recorded.
            'slot' refers to the index of the neuron in the reshaped dataset.
                0 < slot < NUM_NEURONS, the number of neurons in hermaphrodite C. elegans.
        """
        self.worm_dataset = worm_dataset
        self.named_neuron_to_idx = dict()
        self.unknown_neuron_to_idx = dict()
        self.slot_to_named_neuron = dict()
        self.slot_to_unknown_neuron = dict()
        self.slot_to_neuron = dict()
        self.dtype = torch.half
        self._init_neuron_data()
        self._reshape_data()

    def _init_neuron_data(self):
        """Initializes attributes from keys that must already be present in the worm dataset
        at the time that the reshaper is invoked.
        Therefore, those keys specify what is required for a worm dataset to be valid.
        """
        # Post-processed data / absolutely necessary keys
        self.time_in_seconds = self.worm_dataset["time_in_seconds"]
        self.dt = self.worm_dataset["dt"]
        self.max_timesteps = self.worm_dataset["max_timesteps"]
        self.median_dt = self.worm_dataset["median_dt"]
        self.calcium_data = self.worm_dataset["calcium_data"]
        self.smooth_calcium_data = self.worm_dataset["smooth_calcium_data"]
        self.residual_calcium = self.worm_dataset["residual_calcium"]
        self.smooth_residual_calcium = self.worm_dataset["smooth_residual_calcium"]
        self.neuron_to_idx = self.worm_dataset["neuron_to_idx"]
        self.idx_to_neuron = self.worm_dataset["idx_to_neuron"]
        self.extra_info = self.worm_dataset.get("extra_info", dict())
        # Original data / optional keys that may be inferred
        self.original_time_in_seconds = self.worm_dataset.get(
            "original_time_in_seconds", self.worm_dataset["time_in_seconds"]
        )
        self.original_dt = self.worm_dataset.get("original_dt", self.worm_dataset["dt"])
        self.original_max_timesteps = self.worm_dataset.get(
            "original_max_timesteps", self.worm_dataset["max_timesteps"]
        )
        self.original_calcium_data = self.worm_dataset.get(
            "original_calcium_data", self.worm_dataset["calcium_data"]
        )
        self.original_median_dt = self.worm_dataset.get(
            "original_median_dt", self.worm_dataset["median_dt"]
        )
        self.original_smooth_calcium_data = self.worm_dataset.get(
            "original_smooth_calcium_data", self.worm_dataset["smooth_calcium_data"]
        )
        self.original_residual_calcium = self.worm_dataset.get(
            "original_residual_calcium", self.worm_dataset["residual_calcium"]
        )
        self.original_smooth_residual_calcium = self.worm_dataset.get(
            "original_smooth_residual_calcium",
            self.worm_dataset["smooth_residual_calcium"],
        )

    def _reshape_data(self):
        self._prepare_initial_data()
        self._fill_named_neurons_data()
        self._fill_unknown_neurons_data()
        self._update_worm_dataset()
        self._remove_old_mappings()

    def _prepare_initial_data(self):
        assert (
            len(self.idx_to_neuron) == self.calcium_data.shape[1]
        ), "Number of neurons in calcium data matrix does not match number of recorded neurons."
        self.named_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self.unknown_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self._init_empty_calcium_data()
        self._tensor_time_data()

    def _init_empty_calcium_data(self):
        # Resampled data
        self.standard_calcium_data = torch.zeros(self.max_timesteps, NUM_NEURONS, dtype=self.dtype)
        self.standard_residual_calcium = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_smooth_calcium_data = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_residual_smooth_calcium = torch.zeros(
            self.max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        # Raw data
        self.standard_original_calcium_data = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_original_smooth_calcium_data = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_original_residual_calcium = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )
        self.standard_original_smooth_residual_calcium = torch.zeros(
            self.original_max_timesteps, NUM_NEURONS, dtype=self.dtype
        )

    def _tensor_time_data(self):
        # Resampled data
        self.time_in_seconds = (
            self.time_in_seconds - self.time_in_seconds[0]
        )  # start at 0.0 seconds
        self.dt = np.diff(self.time_in_seconds, axis=0, prepend=0.0)
        self.median_dt = np.median(self.dt[1:]).item()
        self.time_in_seconds = torch.from_numpy(self.time_in_seconds).to(self.dtype)
        if self.time_in_seconds.ndim == 1:
            self.time_in_seconds = self.time_in_seconds.unsqueeze(-1)
        self.dt = torch.from_numpy(self.dt).to(self.dtype)
        if self.dt.ndim == 1:
            self.dt = self.dt.unsqueeze(-1)
        # Raw data
        self.original_time_in_seconds = (
            self.original_time_in_seconds - self.original_time_in_seconds[0]
        )  # start at 0.0 seconds
        self.original_dt = np.diff(self.original_time_in_seconds, axis=0, prepend=0.0)
        self.original_median_dt = np.median(self.original_dt[1:]).item()
        self.original_time_in_seconds = torch.from_numpy(self.original_time_in_seconds).to(
            self.dtype
        )
        if self.original_time_in_seconds.ndim == 1:
            self.original_time_in_seconds = self.time_in_seconds.unsqueeze(-1)
        self.original_dt = torch.from_numpy(self.original_dt).to(self.dtype)
        if self.original_dt.ndim == 1:
            self.original_dt = self.original_dt.unsqueeze(-1)

    def _fill_named_neurons_data(self):
        for slot, neuron in enumerate(NEURON_LABELS):
            if neuron in self.neuron_to_idx:  # named neuron
                idx = self.neuron_to_idx[neuron]
                self.named_neuron_to_idx[neuron] = idx
                self._fill_calcium_data(idx, slot)
                self.named_neurons_mask[slot] = True
                self.slot_to_named_neuron[slot] = neuron

    def _fill_calcium_data(self, idx, slot):
        self.standard_calcium_data[:, slot] = torch.from_numpy(self.calcium_data[:, idx]).to(
            self.dtype
        )
        self.standard_residual_calcium[:, slot] = torch.from_numpy(
            self.residual_calcium[:, idx]
        ).to(self.dtype)
        self.standard_smooth_calcium_data[:, slot] = torch.from_numpy(
            self.smooth_calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_residual_smooth_calcium[:, slot] = torch.from_numpy(
            self.smooth_residual_calcium[:, idx]
        ).to(self.dtype)
        # Raw data
        self.standard_original_calcium_data[:, slot] = torch.from_numpy(
            self.original_calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_original_smooth_calcium_data[:, slot] = torch.from_numpy(
            self.original_smooth_calcium_data[:, idx]
        ).to(self.dtype)
        self.standard_original_residual_calcium[:, slot] = torch.from_numpy(
            self.original_residual_calcium[:, idx]
        ).to(self.dtype)
        self.standard_original_smooth_residual_calcium[:, slot] = torch.from_numpy(
            self.original_smooth_residual_calcium[:, idx]
        ).to(self.dtype)

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
                "calcium_data": self.standard_calcium_data,  # normalized, resampled
                "dt": self.dt,  # resampled (vector)
                "idx_to_named_neuron": {v: k for k, v in self.named_neuron_to_idx.items()},
                "idx_to_unknown_neuron": {v: k for k, v in self.unknown_neuron_to_idx.items()},
                "median_dt": self.median_dt,  # resampled (scalar)
                "named_neuron_to_idx": self.named_neuron_to_idx,
                "named_neuron_to_slot": {v: k for k, v in self.slot_to_named_neuron.items()},
                "named_neurons_mask": self.named_neurons_mask,
                "neuron_to_slot": {v: k for k, v in self.slot_to_neuron.items()},
                "neurons_mask": self.named_neurons_mask | self.unknown_neurons_mask,
                "original_calcium_data": self.standard_original_calcium_data,  # original, normalized
                "original_dt": self.original_dt,  # original (vector)
                "original_median_dt": self.original_median_dt,  # original (scalar)
                "original_residual_calcium": self.standard_original_residual_calcium,  # original
                "original_smooth_calcium_data": self.standard_original_smooth_calcium_data,  # original, normalized, smoothed
                "original_smooth_residual_calcium": self.standard_original_smooth_residual_calcium,  # original, smoothed
                "original_time_in_seconds": self.original_time_in_seconds,  # original
                "residual_calcium": self.standard_residual_calcium,  # resampled
                "smooth_calcium_data": self.standard_smooth_calcium_data,  # normalized, smoothed, resampled
                "smooth_residual_calcium": self.standard_residual_smooth_calcium,  # smoothed, resampled
                "slot_to_named_neuron": self.slot_to_named_neuron,
                "slot_to_neuron": self.slot_to_neuron,
                "slot_to_unknown_neuron": self.slot_to_unknown_neuron,
                "time_in_seconds": self.time_in_seconds,  # resampled
                "unknown_neuron_to_idx": self.unknown_neuron_to_idx,
                "unknown_neuron_to_slot": {v: k for k, v in self.slot_to_unknown_neuron.items()},
                "unknown_neurons_mask": self.unknown_neurons_mask,
                "extra_info": self.extra_info,
            }
        )

    def _remove_old_mappings(self):
        keys_to_delete = [key for key in self.worm_dataset if "idx" in key]
        for key in keys_to_delete:
            self.worm_dataset.pop(key, None)


class BasePreprocessor:
    """
    This is a base class used for preprocessing different types of neurophysiological datasets.

    The class provides a template for loading, extracting, smoothing, resampling, and
    normalizing neural data, as well as saving the processed data in a pickle format.
    Specific datasets can be processed by creating a new class that inherits from this base class
    and overriding the methods as necessary.

    Attributes:
        source_dataset (str): The specific source dataset to be preprocessed.
        transform (object): The sklearn transformation to be applied to the data.
        smooth_method (str): The smoothing method to apply to the data.
        resample_dt (float): The resampling time interval in seconds.
        raw_data_path (str): The path where the raw dat is downloaded at.
        processed_data_apth (str): The path at which to save the processed dataset.

    Methods:
        load_data(): Method for loading the raw data.
        extract_data(): Method for extracting the neural data from the raw data.
        smooth_data(): Method for smoothing the neural data.
        resample_data(): Method for resampling the neural data.
        normalize_data(): Method for normalizing the neural data.
        save_data(): Method for saving the processed data to .pickle format.
        create_neuron_idx(): Method for extracting a neuron label to index mapping from the raw data.
        preprocess_traces(): Base method for preprocessing the calcium traces. Some datasets may require
                    additional preprocessing steps, in which case this method should be overridden.

    Note:
        This class is intended to be subclassed, not directly instantiated.
        Specific datasets should implement their own versions of the
        `load_data`,`extract_data`, `smooth_data`, `resample_data`, `normalize_data`,
        `create_metadata` `save_data`, and `preprocess` methods.

    Example:
        class SpecificDatasetPreprocessor(BasePreprocessor):
            def load_data(self):
                # Implement dataset-specific loading logic here.
    """

    def __init__(
        self,
        dataset_name,
        # TODO: Try different transforms from sklearn such as QuantileTransformer, etc. as well as custom CausalNormalizer.
        transform=StandardScaler(),  # StandardScaler() #PowerTransformer() #CausalNormalizer() #None
        smooth_method="ma",
        interpolate_method="linear",
        resample_dt=0.1,
        **kwargs,
    ):
        self.source_dataset = dataset_name
        self.transform = transform
        self.smooth_method = smooth_method
        self.interpolate_method = interpolate_method
        self.resample_dt = resample_dt
        self.smooth_kwargs = kwargs
        self.raw_data_path = os.path.join(ROOT_DIR, "opensource_data")
        self.processed_data_path = os.path.join(ROOT_DIR, "data/processed/neural")

    def smooth_data(self, data, time_in_seconds):
        return smooth_data_preprocess(
            data,
            time_in_seconds,
            self.smooth_method,
            **self.smooth_kwargs,
        )

    def resample_data(self, time_in_seconds, data, upsample=True):
        """
        Args:
            time_in_seconds (np.ndarray): time vector in seconds with shape (time, 1).
            data (np.ndarray): original, non-uniformly sampled calcium data with shape (time, neurons).
            upsample (bool): whether to sample at a higher frequency (i.e with smaller dt).

        Returns:
            np.ndarray, np.ndarray: resampled time vector and calcium data.
        """
        # Upsample (interpolate)
        if upsample:
            return interpolate_data(
                time_in_seconds,
                data,
                target_dt=self.resample_dt,
                method=self.interpolate_method,
            )
        # Downsample (aggregate)
        else:
            # We first upsample to a fraction of the desired dt
            interp_time, interp_ca = interpolate_data(
                time_in_seconds,
                data,
                target_dt=self.resample_dt / 6,
                method=self.interpolate_method,
            )
            # Then average over short intervals to downsample to the desired dt
            return aggregate_data(
                interp_time,
                interp_ca,
                target_dt=self.resample_dt,
            )

    def normalize_data(self, data):
        if self.transform is None:
            return data
        return self.transform.fit_transform(data)

    def save_data(self, data_dict):
        file = os.path.join(self.processed_data_path, f"{self.source_dataset}.pickle")
        with open(file, "wb") as f:
            pickle.dump(data_dict, f)

    def create_neuron_idx(self, unique_IDs):
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(unique_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "") if not name.endswith("0") and not name.isnumeric() else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = {
            nid: (str(nid) if name not in set(NEURON_LABELS) else name)
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        # Number of neurons that were labeled with a name
        num_named_neurons = len([k for k in neuron_to_idx.keys() if not k.isnumeric()])
        return neuron_to_idx, num_named_neurons

    def load_data(self, file_name):
        """
        A simple place-holder method for loading raw data from a .mat file,
        that works for the Skora, Kato, Nichols, Uzel, and Kaplan datasets.
        """
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))

    def extract_data(self):
        """
        Extracts the basic data from the raw data file,
        which is the neuron IDs, calcium traces, and time vector.
        """
        raise NotImplementedError()

    def create_metadata(self):
        """
        Place-holder method for making dictionary of
        extra information or metadata for a dataset.
        """
        extra_info = dict()
        return extra_info

    def pick_non_none(self, l):
        """
        Returns the first non-None element in a list, l.
        """
        for i in range(len(l)):
            if l[i] is not None:
                return l[i]
        return None

    def preprocess(self):
        """
        Main preprocessing method that calls the other methods in the class.
        Preprocesses the calcium imaging data for all worms and packages into
        a single dataset caontaining data multiple worms.
        """
        raise NotImplementedError()

    def preprocess_traces(
        self,
        neuron_IDs,
        traces,
        raw_timeVectorSeconds,
        preprocessed_data,
        worm_idx,
    ):
        """
        Helper function for preprocessing calcium fluorescence neural data from one worm.

        Args:
            neuron_IDs (list): List of arrays of neuron IDs.
            traces (list): List of arrays of calcium traces, with indices corresponding to neuron_IDs.
            raw_timeVectorSeconds (list): List of arrays of time vectors, with indices corresponding to neuron_IDs.
            preprocessed_data (dict): Dictionary of preprocessed data from previous worms that gets extended with more worms here.
            worm_idx (int): Index of the current worm.

        Returns:
            dict: Collection of all preprocessed worm data so far.
            int: Index of the next worm to preprocess.
        """
        for i, trace_data in enumerate(traces):
            # Matrix `trace_data` should be shaped as (time, neurons)
            assert trace_data.ndim == 2, "Calcium traces must be 2D arrays."
            assert trace_data.shape[1] == len(
                neuron_IDs[i]
            ), "Calcium trace does not have the right number of neurons."
            # 0. Ignore any worms with empty traces and name worm
            if trace_data.size == 0:
                continue
            # 1. Map named neurons
            unique_IDs = [
                (self.pick_non_none(j) if isinstance(j, list) else j) for j in neuron_IDs[i]
            ]
            unique_IDs = [
                (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                for _, j in enumerate(unique_IDs)
            ]
            _, unique_indices = np.unique(unique_IDs, return_index=True)
            unique_IDs = [unique_IDs[_] for _ in unique_indices]
            # Create neuron label to index mapping
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)
            # Skip worms with no labelled neurons
            if num_named_neurons == 0:
                continue
            # Only get data for unique neurons
            trace_data = trace_data[:, unique_indices.astype(int)]
            # 2. Normalize calcium data
            calcium_data = self.normalize_data(trace_data)  # matrix
            # 3. Compute calcium dynamics (residual calcium)
            time_in_seconds = raw_timeVectorSeconds[i].reshape(raw_timeVectorSeconds[i].shape[0], 1)
            time_in_seconds = np.array(time_in_seconds, dtype=np.float32)  # vector
            time_in_seconds = time_in_seconds - time_in_seconds[0]  # start at 0.0 seconds
            dt = np.diff(time_in_seconds, axis=0, prepend=0.0)  # vector
            original_median_dt = np.median(dt[1:]).item()  # scalar
            residual_calcium = np.gradient(
                calcium_data, time_in_seconds.squeeze(), axis=0
            )  # vector
            # 4. Smooth data
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(residual_calcium, time_in_seconds)
            # 5. Resample data (raw and smoothed data)
            upsample = self.resample_dt < original_median_dt  # bool
            _, resampled_calcium_data = self.resample_data(time_in_seconds, calcium_data, upsample)
            _, resampled_residual_calcium = self.resample_data(
                time_in_seconds, residual_calcium, upsample
            )
            # NOTE: We use the resampling of the smooth calcium data to give us the resampled time points
            resampled_time_in_seconds, resampled_smooth_calcium_data = self.resample_data(
                time_in_seconds, smooth_calcium_data, upsample
            )
            resampled_time_in_seconds = (
                resampled_time_in_seconds - resampled_time_in_seconds[0]
            )  # start at 0.0 seconds
            _, resampled_smooth_residual_calcium = self.resample_data(
                time_in_seconds, smooth_residual_calcium, upsample
            )
            resampled_dt = np.diff(resampled_time_in_seconds, axis=0, prepend=0.0)  # vector
            resampled_median_dt = np.median(resampled_dt[1:]).item()  # scalar
            assert np.isclose(
                self.resample_dt, resampled_median_dt, atol=0.01
            ), "Resampling failed."
            max_timesteps, num_neurons = resampled_calcium_data.shape
            num_unknown_neurons = int(num_neurons) - num_named_neurons
            # 6. Name worm and update index
            worm = "worm" + str(worm_idx)  # use global worm index
            worm_idx += 1  # increment worm index
            # 7. Save data
            worm_dict = {
                worm: {
                    "calcium_data": resampled_calcium_data,  # normalized and resampled
                    "source_dataset": self.source_dataset,
                    "dt": resampled_dt,  # vector from resampled time vector
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "interpolate_method": self.interpolate_method,
                    "max_timesteps": int(max_timesteps),  # scalar from resampled time vector
                    "median_dt": self.resample_dt,  # scalar from resampled time vector
                    "neuron_to_idx": neuron_to_idx,
                    "num_named_neurons": num_named_neurons,
                    "num_neurons": int(num_neurons),
                    "num_unknown_neurons": num_unknown_neurons,
                    "original_dt": dt,  # vector from original time vector
                    "original_calcium_data": calcium_data,  # normalized
                    "original_max_timesteps": int(
                        calcium_data.shape[0]
                    ),  # scalar from original time vector
                    "original_median_dt": original_median_dt,  # scalar from original time vector
                    "original_residual_calcium": residual_calcium,  # original
                    "original_smooth_calcium_data": smooth_calcium_data,  # normalized and smoothed
                    "original_smooth_residual_calcium": smooth_residual_calcium,  # smoothed
                    "original_time_in_seconds": time_in_seconds,  # original time vector
                    "residual_calcium": resampled_residual_calcium,  # resampled
                    "smooth_calcium_data": resampled_smooth_calcium_data,  # normalized, smoothed and resampled
                    "smooth_method": self.smooth_method,
                    "smooth_residual_calcium": resampled_smooth_residual_calcium,  # smoothed and resampled
                    "time_in_seconds": resampled_time_in_seconds,  # resampled time vector
                    "worm": worm,  # worm ID
                    "extra_info": self.create_metadata(),  # additional information and metadata
                }
            }
            # Update preprocessed data collection
            preprocessed_data.update(worm_dict)
        # Return the updated preprocessed data and worm index
        return preprocessed_data, worm_idx


class Kato2015Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Kato2015",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, arr):
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"] if "IDs" in arr.keys() else arr["NeuronNames"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"] if "traces" in arr.keys() else arr["deltaFOverF_bc"]
        # Time vector in seconds
        timeVectorSeconds = (
            arr["timeVectorSeconds"] if "timeVectorSeconds" in arr.keys() else arr["tv"]
        )
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        extra_info = dict(
            citation="Kato et al., Cell 2015, _Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans_"
        )
        return extra_info

    def preprocess(self):
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Load and preprocess data
        for file_name in ["WT_Stim.mat", "WT_NoStim.mat"]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs, traces, raw_timeVectorSeconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Nichols2017Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Nichols2017",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, arr):
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]
        # Time vector in seconds
        timeVectorSeconds = arr["timeVectorSeconds"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        extra_info = dict(
            citation="Nichols et al., Science 2017, _A global brain state underlies C. elegans sleep behavior_"
        )
        return extra_info

    def preprocess(self):
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Each file contains data of many worms under the same experiment condition
        for file_name in [
            "n2_let.mat",  # let = lethargus = late larval stage 4 (L4)
            "n2_prelet.mat",  # n2 = standard lab strain, more solitary
            "npr1_let.mat",  # npr-1 = proxy for wild-type strain, more social
            "npr1_prelet.mat",  # prelet = pre-lethargus = mid-L4 stage
        ]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs, traces, raw_timeVectorSeconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Skora2018Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Skora2018",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, arr):
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]
        # Time vector in seconds
        timeVectorSeconds = arr["timeVectorSeconds"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        extra_info = dict(
            citation="Skora et al., Cell Reports 2018, _Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans_"
        )
        return extra_info

    def preprocess(self):
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Load and preprocess data
        for file_name in ["WT_fasted.mat", "WT_starved.mat"]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs, traces, raw_timeVectorSeconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Kaplan2020Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Kaplan2020",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        # Load data with mat73
        data = mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, arr):
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["neuron_ID"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces_bleach_corrected"]
        # Time vector in seconds
        timeVectorSeconds = arr["time_vector"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        extra_info = dict(
            citation="Kaplan et al., Neuron 2020, _Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales_"
        )
        return extra_info

    def preprocess(self):
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Each file contains data of many worms under the same experiment condition
        for file_name in [
            "Neuron2019_Data_MNhisCl_RIShisCl.mat",  # MN = motor neuron; RIS = quiescence-promoting neuron
            "Neuron2019_Data_RIShisCl.mat",  # SMD = excitatory motor neurons targeting head and neck muscle
            "Neuron2019_Data_SMDhisCl_RIShisCl.mat",  # hisCL = histamine-gated chloride channel (inhibitory)
        ]:
            data_key = "_".join((file_name.split(".")[0].strip("Neuron2019_Data_"), "Neuron2019"))
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs, traces, raw_timeVectorSeconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Yemini2021Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Yemini2021",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        # Overriding the base class method to use scipy.io.loadmat for .mat files
        data = loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, raw_data):
        """
        A more complicated `extract_data` method
        was necessary for the Yemini2021 dataset.
        """
        # Frames per second
        fps = raw_data["fps"].item()
        # There several files (each is data for one worm) in each .mat file
        files = [_.item() for _ in raw_data["files"].squeeze()]
        # The list `bilat_neurons` does not disambiguate L/R neurons, so we need to do that
        bilat_neurons = [_.item() for _ in raw_data["neurons"].squeeze()]
        # List of lists. Outer list same length as `neuron`. Inner lists are boolean masks for L/R neurons organized by file in `files`.
        is_left_neuron = [  # in each inner list, all L (1) neurons appear before all R (0) neurons
            _.squeeze().tolist() for _ in raw_data["is_L"].squeeze()
        ]  # non-bilateral neurons are nan
        # Histogram-normalized neuronal traces linearly scaled and offset so that neurons are comparable
        norm_traces = [
            _.squeeze().tolist() for _ in raw_data["norm_traces"].squeeze()
        ]  # list-of-lists like `is_left_neuron`
        # This part is the meat of the `extract_data` method
        neuron_IDs = []
        traces = []
        time_vector_seconds = []
        # Each file contains data for one worm
        for f, file in enumerate(files):
            neurons = []
            activity = []
            tvec = np.empty(0, dtype=np.float32)
            for i, neuron in enumerate(bilat_neurons):
                # Assign neuron names with L/R and get associated traces
                bilat_bools = is_left_neuron[i]  # tells us if neuron is L/R
                bilat_traces = norm_traces[i]
                assert len(bilat_traces) == len(
                    bilat_bools
                ), f"Something is wrong with the data. Traces don't match with bilateral mask: {len(bilat_traces)} != {len(bilat_bools)}"
                righty = None
                if len(bilat_bools) // len(files) == 2:
                    # Get lateral assignment
                    lefty = bilat_bools[: len(bilat_bools) // 2][f]
                    righty = bilat_bools[len(bilat_bools) // 2 :][f]
                    # Get traces
                    left_traces = bilat_traces[: len(bilat_traces) // 2][f]
                    right_traces = bilat_traces[len(bilat_traces) // 2 :][f]
                elif len(bilat_bools) == len(files):
                    # Get lateral assignment
                    lefty = bilat_bools[:][f]
                    righty = None
                    # Get traces
                    left_traces = bilat_traces[:][f]
                    right_traces = None
                else:
                    raise ValueError(
                        f"Something is wrong with the data.\nNeuron: {neuron}. File: {file}."
                    )
                if np.isnan(lefty):  # non-bilaterally symmetric neuron
                    act = bilat_traces[f].squeeze().astype(float)
                    neurons.append(None if act.size == 0 else f"{neuron}")
                    activity.append(act)
                else:
                    if lefty == 1:  # left neuron
                        act = left_traces.squeeze().astype(float)
                        neurons.append(None if act.size == 0 else f"{neuron}L")
                        activity.append(act)
                    if righty != None:  # right neuron
                        act = right_traces.squeeze().astype(float)
                        tvec = np.arange(act.size) / fps
                        neurons.append(None if act.size == 0 else f"{neuron}R")
                        activity.append(act)
                # Deal with  time vector which should be the same across all neurons
                if act.size > 0 and act.size > tvec.size:
                    tvec = np.arange(act.size) / fps
            # Add neurons to list of neuron_IDs
            neuron_IDs.append(neurons)
            # Reshape activity to be a 2D array with shape (time, neurons)
            activity = np.stack(
                [
                    np.zeros_like(tvec, dtype=np.float32) if act.size == 0 else act
                    for act in activity
                ],
                dtype=np.float32,
            ).T  # (time, neurons)
            # Impute any remaining NaN values
            imputer = IterativeImputer(random_state=0)
            if np.isnan(activity).any():
                activity = imputer.fit_transform(activity)
            # Add acitvity to list of traces
            traces.append(activity)
            # Add time vector to list of time vectors
            time_vector_seconds.append(tvec)
        # Return the extracted data
        return neuron_IDs, traces, time_vector_seconds

    def create_metadata(self):
        extra_info = dict(
            citation="Yemini et al., Cell CurrBio 2022, _NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans_"
        )
        return extra_info

    def preprocess(self):
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Multiple .mat files to iterate over
        for file_name in [
            "Head_Activity_OH15500.mat",
            "Head_Activity_OH16230.mat",
            "Tail_Activity_OH16230.mat",
        ]:
            raw_data = self.load_data(file_name)  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs, traces, raw_timeVectorSeconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Uzel2022Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Uzel2022",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        # Load data with mat73
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))

    def extract_data(self, arr):
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]  # (time, neurons)
        # Time vector in seconds
        timeVectorSeconds = arr["tv"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        extra_info = dict(
            citation="Uzel et al., Cell CurrBio 2022, _A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans_"
        )
        return extra_info

    def preprocess(self):
        preprocessed_data = dict()  # Store preprocessed data
        worm_idx = 0  # Initialize worm index outside file loop
        # Load and preprocess data
        for file_name in ["Uzel_WT.mat"]:
            data_key = "Uzel_WT"
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs, traces, raw_timeVectorSeconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Lin2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Lin2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        # Overriding the base class method to use scipy.io.loadmat for .mat files
        data = loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, data_file):
        """Slightly different extract_data method for Lin2023 dataset."""
        dataset_raw = self.load_data(data_file)
        # Filter for proofread neurons.
        _filter = dataset_raw["use_flag"].flatten() > 0
        neurons = [str(_.item()) for _ in dataset_raw["proofread_neurons"].flatten()[_filter]]
        raw_time_vec = np.array(dataset_raw["times"].flatten()[0][-1])
        raw_activitiy = dataset_raw["corrected_F"][_filter].T  # (time, neurons)
        # Replace first nan with F0 value
        _f0 = dataset_raw["F_0"][_filter][:, 0]
        raw_activitiy[0, :] = _f0
        # Impute any remaining NaN values
        imputer = IterativeImputer(random_state=0)
        if np.isnan(raw_activitiy).any():
            raw_activitiy = imputer.fit_transform(raw_activitiy)
        # Make the extracted data into a list of lists
        neuron_IDs, raw_traces, time_vector_seconds = [neurons], [raw_activitiy], [raw_time_vec]
        # Return the extracted data
        return neuron_IDs, raw_traces, time_vector_seconds

    def create_metadata(self):
        extra_info = dict(
            citation="Lin et al., Science Advances 2023, _Functional Imaging and Quantification of Multineuronal Olfactory Responses in C. Elegans_"
        )
        return extra_info

    def preprocess(self):
        # Load and preprocess data
        preprocessed_data = dict()  # Store preprocessed data
        worm_idx = 0  # Initialize worm index outside file loop
        # Have multiple .mat files that you iterate over
        data_files = os.path.join(self.raw_data_path, "Lin2023")
        # Multiple .mat files to iterate over
        for file in os.listdir(data_files):
            if not file.endswith(".mat"):
                continue
            neurons, raw_traces, time_vector_seconds = self.extract_data(file)
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons, raw_traces, time_vector_seconds, preprocessed_data, worm_idx
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")
        return None


class Leifer2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Leifer2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def str_to_float(self, str_num):
        """
        Helper function for changingin textual scientific
        notation into a floating-point number.
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
        """ "Helper function to load neuron labels from text file."""
        with open(file_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines()]
        return labels

    def load_time_vector(self, file_path):
        """ "Helper function to load time vector from text file."""
        with open(file_path, "r") as f:
            timeVectorSeconds = [self.str_to_float(line.strip("\n")) for line in f.readlines()]
            timeVectorSeconds = np.array(timeVectorSeconds, dtype=np.float32).reshape(-1, 1)
        return timeVectorSeconds

    def load_data(self, file_name):
        with open(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r") as f:
            data = [list(map(float, line.split(" "))) for line in f.readlines()]
        data_array = np.array(data, dtype=np.float32)
        return data_array

    def create_neuron_idx(self, label_list):
        """
        Overrides the base class method to handle the complicated data
        format structure of the Leifer2023 dataset.
        """
        neuron_to_idx = dict()
        num_unnamed_neurons = 0
        for j, item in enumerate(label_list):
            previous_list = label_list[:j]
            if not item.isalnum():
                label_list[j] = str(j)
                num_unnamed_neurons += 1
                neuron_to_idx[str(j)] = j
            else:
                if item in NEURON_LABELS and item not in previous_list:
                    neuron_to_idx[item] = j
                elif item in NEURON_LABELS and item in previous_list:
                    label_list[j] = str(j)
                    num_unnamed_neurons += 1
                    neuron_to_idx[str(j)] = j
                else:
                    if str(item + "L") in NEURON_LABELS and str(item + "L") not in previous_list:
                        label_list[j] = str(item + "L")
                        neuron_to_idx[str(item + "L")] = j
                    elif str(item + "R") in NEURON_LABELS and str(item + "R") not in previous_list:
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
        ), "Incorrect calculation of the number of named neurons."
        return neuron_to_idx, num_named_neurons

    def extract_data(self, data_file, labels_file, time_file):
        """Slightly different `extract_data` needed method for Leifer2023 dataset."""
        real_data = self.load_data(data_file)
        label_list = self.load_labels(labels_file)[: real_data.shape[1]]
        time_in_seconds = self.load_time_vector(time_file)
        # Remove columns where all values are NaN
        mask = np.argwhere(~np.isnan(real_data).all(axis=0)).flatten()
        real_data = real_data[:, mask]
        label_list = np.array(label_list, dtype=str)[mask].tolist()
        # Impute any remaining NaN values
        imputer = IterativeImputer(random_state=0)
        if np.isnan(real_data).any():
            real_data = imputer.fit_transform(real_data)
        # Check that the data and labels match
        assert real_data.shape[1] == len(
            label_list
        ), "Data and labels do not match!\n Files: {data_file}, {labels_file}"
        assert (
            real_data.shape[0] == time_in_seconds.shape[0]
        ), "Time vector does not match data!\n Files: {data_file}, {time_file}"
        # Return the extracted data
        return label_list, real_data, time_in_seconds

    def create_metadata(self):
        extra_info = dict(
            citation="Randi et al., Nature 2023, _Neural Signal Propagation Atlas of Caenorhabditis Elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        The `preprocess` method for the Leifer 2023 dataset is significantly different
        than that for the other datasets of differences between the file structure containing
        the raw data for the Leifer2023 dataset compared to the other source datasets:
            - Leifer2023 raw data uses 6 files per worm each containing distinct information.
            - The other datasets use 1 file containing all the information for multiple worms.
        Unlike the `preprocess` method in the other dataset classes which makes use of the
        `preprocess_traces` method from the parent BasePreprocessor class, this one does not.
        """
        # TODO: Encapsulate the single worm part of this method into a `preprocess_traces` method.
        # Load and preprocess data
        preprocessed_data = dict()
        data_dir = os.path.join(self.raw_data_path, self.source_dataset)
        # Every worm has 6 text files
        files = os.listdir(data_dir)
        num_worms = int(len(files) / 6)
        # Initialize worm index outside file loop
        worm_idx = 0
        # Iterate over each worm's data text files
        for i in range(0, num_worms):
            worm = f"worm{str(worm_idx)}"
            worm_idx += 1
            data_file = os.path.join(data_dir, f"{str(i)}_gcamp.txt")
            labels_file = os.path.join(data_dir, f"{str(i)}_labels.txt")
            time_file = os.path.join(data_dir, f"{str(i)}_t.txt")
            # Load and extract raw data
            label_list, real_data, time_in_seconds = self.extract_data(
                data_file, labels_file, time_file
            )
            # Set time to start at 0.0 seconds
            time_in_seconds = time_in_seconds - time_in_seconds[0]  # vector
            # Skip worms with no recorded neurons
            if len(label_list) == 0:
                worm_idx -= 1
                continue
            # Skip worms with very short recordings
            if len(time_in_seconds) < 700:
                worm_idx -= 1
                continue
            # 1. Map named neurons
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(label_list)
            if num_named_neurons == 0:  # skip worms with no labelled neuron
                worm_idx -= 1
                continue
            # 2. Normalize calcium data
            calcium_data = self.normalize_data(real_data)  # matrix
            # 3. Compute calcium dynamics (residual calcium)
            dt = np.diff(time_in_seconds, axis=0, prepend=0.0)  # vector
            original_median_dt = np.median(dt[1:]).item()  # scalar
            residual_calcium = np.gradient(
                calcium_data, time_in_seconds.squeeze(), axis=0
            )  # vector
            # 4. Smooth data
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(residual_calcium, time_in_seconds)
            # 5. Resample data (raw and smoothed data)
            upsample = original_median_dt >= self.resample_dt  # bool: whether to up/down-sample
            _, resampled_calcium_data = self.resample_data(time_in_seconds, calcium_data, upsample)
            _, resampled_residual_calcium = self.resample_data(
                time_in_seconds, residual_calcium, upsample
            )
            # NOTE: We use the resampling of the smooth calcium data to give us the resampled time points
            resampled_time_in_seconds, resampled_smooth_calcium_data = self.resample_data(
                time_in_seconds, smooth_calcium_data, upsample
            )
            resampled_time_in_seconds = (
                resampled_time_in_seconds - resampled_time_in_seconds[0]
            )  # start at 0.0 seconds
            _, resampled_smooth_residual_calcium = self.resample_data(
                time_in_seconds, smooth_residual_calcium, upsample
            )
            resampled_dt = np.diff(resampled_time_in_seconds, axis=0, prepend=0.0)  # vector
            resampled_median_dt = np.median(resampled_dt[1:]).item()  # scalar
            assert np.isclose(self.resample_dt, resampled_median_dt), "Resampling failed."
            max_timesteps, num_neurons = resampled_calcium_data.shape
            num_unknown_neurons = int(num_neurons) - num_named_neurons
            # 6. Save data
            worm_dict = {
                worm: {
                    "calcium_data": resampled_calcium_data,  # normalized and resampled
                    "source_dataset": self.source_dataset,
                    "dt": resampled_dt,  # vector from resampled time vector
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "interpolate_method": self.interpolate_method,
                    "max_timesteps": int(max_timesteps),  # scalar from resampled time vector
                    "median_dt": self.resample_dt,  # scalar from resampled time vector
                    "neuron_to_idx": neuron_to_idx,
                    "num_named_neurons": num_named_neurons,
                    "num_neurons": int(num_neurons),
                    "num_unknown_neurons": num_unknown_neurons,
                    "original_calcium_data": calcium_data,  # normalized
                    "original_max_timesteps": int(
                        calcium_data.shape[0]
                    ),  # scalar from original time vector
                    "original_dt": dt,  # vector from original time vector
                    "original_median_dt": original_median_dt,  # scalar from original time vector
                    "original_residual_calcium": residual_calcium,  # original
                    "original_smooth_calcium_data": smooth_calcium_data,  # normalized and smoothed
                    "original_smooth_residual_calcium": smooth_residual_calcium,  # smoothed
                    "original_time_in_seconds": time_in_seconds,  # original time vector
                    "residual_calcium": resampled_residual_calcium,  # resampled
                    "smooth_calcium_data": resampled_smooth_calcium_data,  # normalized, smoothed and resampled
                    "smooth_method": self.smooth_method,
                    "smooth_residual_calcium": resampled_smooth_residual_calcium,  # smoothed and resampled
                    "time_in_seconds": resampled_time_in_seconds,  # resampled time vector
                    "worm": worm,  # worm ID
                    "extra_info": self.create_metadata(),  # additional information and metadata
                }
            }
            # Update preprocessed data collection
            preprocessed_data.update(worm_dict)
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Flavell2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        super().__init__(
            "Flavell2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def find_nearest_label(self, query, possible_labels, char="?"):
        """Find the nearest neuron label from a list given a query."""
        # Remove the '?' from the query to simplify comparison
        query_base = query.replace(char, "")
        # Initialize variables to track the best match
        nearest_label = None
        highest_similarity = -1  # Start with lowest similarity possible
        for label in possible_labels:
            # Count matching characters, ignoring the character at the position of '?'
            similarity = sum(1 for q, l in zip(query_base, label) if q == l)
            # Update the nearest label if this one is more similar
            if similarity > highest_similarity:
                nearest_label = label
                highest_similarity = similarity
        return nearest_label, possible_labels.index(nearest_label)

    def load_data(self, file_name):
        if file_name.endswith(".h5"):
            data = h5py.File(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r")
        elif file_name.endswith(".json"):
            with open(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
        return data

    def extract_data(self, file_data):
        """
        A more complicated `extract_data` method was necessary for the Flavell2023 dataset.
        """
        # If files use H5 format
        if isinstance(file_data, h5py.File):
            # Time vector in seconds
            time_in_seconds = np.array(file_data["timestamp_confocal"], dtype=np.float32)
            time_in_seconds = time_in_seconds.reshape((-1, 1))
            calcium_data = np.array(file_data["trace_array"], dtype=np.float32)
            neurons = np.array(file_data["neuropal_label"], dtype=str)
            # For bilateral neurons with ambiguous location, we randomly assign L/R
            neurons_copy = []
            # TODO: This is not comprehensive enough; make more like below
            for neuron in neurons:
                if neuron.replace("?", "L") not in set(neurons_copy):
                    neurons_copy.append(neuron.replace("?", "L"))
                else:
                    neurons_copy.append(neuron.replace("?", "R"))
            # Filter for unique neuron labels
            neurons = np.array(neurons_copy, dtype=str)
            neurons, unique_indices = np.unique(neurons, return_index=True, return_counts=False)
            # Only get data for unique neurons
            calcium_data = calcium_data[:, unique_indices]
        # Otherwise if files use JSON format
        elif isinstance(file_data, dict):
            # Time vector in seconds
            time_in_seconds = np.array(file_data["timestamp_confocal"], dtype=np.float32)
            time_in_seconds = time_in_seconds.reshape((-1, 1))
            # Raw traces (list)
            raw_traces = file_data["trace_array"]
            # Max time steps (int)
            max_t = len(raw_traces[0])
            # Number of neurons (int)
            number_neurons = len(raw_traces)
            # Labels (list)
            ids = file_data["labeled"]
            # All traces
            calcium_data = np.zeros((max_t, number_neurons), dtype=np.float32)
            for i, trace in enumerate(raw_traces):
                calcium_data[:, i] = trace
            neurons = [str(i) for i in range(number_neurons)]
            for i in ids.keys():
                label = ids[str(i)]["label"]
                neurons[int(i) - 1] = label
            # Handle ambiguous neuron labels
            for i in range(number_neurons):
                label = neurons[i]
                if not label.isnumeric():
                    # Treat the '?' in labels
                    if "?" in label and "??" not in label:
                        # Find the group which the neuron belongs to
                        label_split = label.split("?")
                        # Find potential label matches excluding ones we already have
                        possible_labels = [
                            neuron_name
                            for neuron_name in NEURON_LABELS
                            if (label_split[0] in neuron_name)
                            and (label_split[-1] in neuron_name)
                            and (neuron_name not in set(neurons))
                        ]
                        # Pick the neuron label with the nearest similarity
                        neuron_label, _ = self.find_nearest_label(label, possible_labels, char="?")
                        neurons[i] = neuron_label
                    # Treat the '??' in labels
                    elif "??" in label:
                        # Find the group which the neuron belongs to
                        label_split = label.split("??")
                        # Find potential label matches excluding ones we already have
                        possible_labels = [
                            neuron_name
                            for neuron_name in NEURON_LABELS
                            if (label_split[0] in neuron_name)
                            and (label_split[-1] in neuron_name)
                            and (neuron_name not in set(neurons))
                        ]
                        # Pick the neuron label with the nearest similarity
                        neuron_label, _ = self.find_nearest_label(label, possible_labels, char="??")
                        neurons[i] = neuron_label
            # Filter for unique neuron labels
            neurons = np.array(neurons, dtype=str)
            neurons, unique_indices = np.unique(neurons, return_index=True, return_counts=False)
            # Only get data for unique neurons
            calcium_data = calcium_data[:, unique_indices]
        else:
            raise ValueError(f"Unsupported data type: {type(file_data)}")
        # Return the extracted data
        return neurons, calcium_data, time_in_seconds

    def create_metadata(self):
        extra_info = dict(
            citation="Atanas et al., Cell 2023, _Brain-Wide Representations of Behavior Spanning Multiple Timescales and States in C. Elegans_"
        )
        return extra_info

    def preprocess(self):
        # TODO: Encapsulate the single worm part of this method into a `preprocess_traces` method.
        # Load and preprocess data
        preprocessed_data = dict()
        for i, file in enumerate(os.listdir(os.path.join(self.raw_data_path, self.source_dataset))):
            if not (file.endswith(".h5") or file.endswith(".json")):
                continue
            worm = "worm" + str(i)
            file_data = self.load_data(file)  # load
            # 0. Extract raw data
            neurons, calcium_data, time_in_seconds = self.extract_data(file_data)
            # Set time to start at 0.0 seconds
            time_in_seconds = time_in_seconds - time_in_seconds[0]  # vector
            # 1. Map named neurons
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(neurons)
            # 2. Normalize calcium data
            calcium_data = self.normalize_data(calcium_data)  # matrix
            # 3. Compute calcium dynamics (residual calcium)
            dt = np.diff(time_in_seconds, axis=0, prepend=0.0)  # vector
            original_median_dt = np.median(dt[1:]).item()  # scalar
            residual_calcium = np.gradient(
                calcium_data, time_in_seconds.squeeze(), axis=0
            )  # vector
            # 4. Smooth data
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(residual_calcium, time_in_seconds)
            # 5. Resample data (raw and smoothed data)
            upsample = original_median_dt >= self.resample_dt  # bool: whether to up/down-sample
            _, resampled_calcium_data = self.resample_data(time_in_seconds, calcium_data, upsample)
            _, resampled_residual_calcium = self.resample_data(
                time_in_seconds, residual_calcium, upsample
            )
            # NOTE: We use the resampling of the smooth calcium data to give us the resampled time points
            resampled_time_in_seconds, resampled_smooth_calcium_data = self.resample_data(
                time_in_seconds, smooth_calcium_data, upsample
            )
            resampled_time_in_seconds = (
                resampled_time_in_seconds - resampled_time_in_seconds[0]
            )  # start at 0.0 seconds
            _, resampled_smooth_residual_calcium = self.resample_data(
                time_in_seconds, smooth_residual_calcium, upsample
            )
            resampled_dt = np.diff(resampled_time_in_seconds, axis=0, prepend=0.0)  # vector
            resampled_median_dt = np.median(resampled_dt[1:]).item()  # scalar
            assert np.isclose(self.resample_dt, resampled_median_dt), "Resampling failed."
            max_timesteps, num_neurons = resampled_calcium_data.shape
            num_unknown_neurons = int(num_neurons) - num_named_neurons
            # 6. Save data
            worm_dict = {
                worm: {
                    "calcium_data": resampled_calcium_data,  # normalized and resampled
                    "source_dataset": self.source_dataset,
                    "dt": resampled_dt,  # vector from resampled time vector
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "interpolate_method": self.interpolate_method,
                    "max_timesteps": int(max_timesteps),  # scalar from resampled time vector
                    "median_dt": self.resample_dt,  # scalar from resampled time vector
                    "neuron_to_idx": neuron_to_idx,
                    "num_named_neurons": num_named_neurons,
                    "num_neurons": int(num_neurons),
                    "num_unknown_neurons": num_unknown_neurons,
                    "original_dt": dt,  # vector from original time vector
                    "original_calcium_data": calcium_data,  # normalized
                    "original_max_timesteps": int(
                        calcium_data.shape[0]
                    ),  # scalar from original time vector
                    "original_median_dt": original_median_dt,  # scalar from original time vector
                    "original_residual_calcium": residual_calcium,  # original
                    "original_smooth_calcium_data": smooth_calcium_data,  # normalized and smoothed
                    "original_smooth_residual_calcium": smooth_residual_calcium,  # smoothed
                    "original_time_in_seconds": time_in_seconds,  # original time vector
                    "residual_calcium": resampled_residual_calcium,  # resampled
                    "smooth_calcium_data": resampled_smooth_calcium_data,  # normalized, smoothed and resampled
                    "smooth_method": self.smooth_method,
                    "smooth_residual_calcium": resampled_smooth_residual_calcium,  # smoothed and resampled
                    "time_in_seconds": resampled_time_in_seconds,  # resampled time vector
                    "worm": worm,  # worm ID
                    "extra_info": self.create_metadata(),  # additional information and metadata
                }
            }
            # Update preprocessed data collection
            preprocessed_data.update(worm_dict)
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
