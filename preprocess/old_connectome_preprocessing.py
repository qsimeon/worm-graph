from preprocess._pkg import *

# TODO: Encapsulate the connectome preprocessing helper functions into a class (ConnectomeBasePreprocessor) much like we did with neural activity processing.


def preprocess_connectome(raw_dir, raw_files, pub="witvliet_7"):
    """Convert the raw connectome data to a graph tensor.

    This function processes raw connectome data, which includes chemical
    synapses and gap junctions, into a format suitable for use in machine
    learning or graph analysis. It reads the raw data in .csv format,
    processes it to extract relevant information, and creates graph
    tensors that represent the C. elegans connectome. The resulting
    graph tensors are saved in the 'data/processed/connectome' folder
    as 'graph_tensors.pt'. We distinguish between electrical (gap junction)
    and chemical synapses by using an edge attribute tesnor with two feature dimensions:
    the first feature represents the weight of the gap junctions; and the second feature
    represents the weight of the chemical synapses.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data
    raw_files : list
        Contain the names of the raw connectome data to preprocess
    pub : str, optional
        The publication to use for preprocessing. Options include:
        - "openworm": OpenWorm project
        - "funconn" or "randi_2023": Randi et al., 2023 (functional connectivity)
        - "witvliet_7" (default): Witvliet et al., 2020 (adult 7)
        - "witvliet_8": Witvliet et al., 2020 (adult 8)
        - "white_1986_whole": White et al., 1986 (whole)
        - "white_1986_n2u": White et al., 1986 (N2U)
        - "white_1986_jsh": White et al., 1986 (JSH)
        - "white_1986_jse": White et al., 1986 (JSE)
        - "cook_2019": Cook et al., 2019
        - None: Default to a preprocessed variant of Cook et al., 2019

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
    * The default connectome data used here is from Cook et al., 2019.
      If the raw data isn't found, please download the zip file from this link:
      https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip,
      unzip the archive in the data/raw folder, then run the MATLAB script `export_nodes_edges.m`.
    """
    # Check if the raw connectome data exists
    if not os.path.exists(raw_dir):
        download_url(url=RAW_DATA_URL, folder=ROOT_DIR, filename=RAW_ZIP)
        extract_zip(
            path=os.path.join(ROOT_DIR, RAW_ZIP),
            folder=RAW_DATA_DIR,
            delete_zip=True,
        )
        raw_dir = RAW_DATA_DIR

    # Check that all the necessary raw files were extracted
    assert all(
        [os.path.exists(os.path.join(raw_dir, rf)) for rf in raw_files]
    ), f"Some necessary connectome data files were not found in {raw_dir}."

    # Determine appropriate preprocessing function based on publication
    if pub == "openworm":
        preprocess_openworm(raw_dir)
    elif pub == "funconn" or pub == "randi_2023":
        preprocess_randi_2023_funconn(raw_dir)
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
    else:  # if pub is None
        preprocess_default(raw_dir)
        pass

    return None


def preprocess_common_tasks(raw_dir, edge_index, edge_attr):
    # Load the neuron master sheet
    df_master = pd.read_csv(os.path.join(raw_dir, "neuron_master_sheet.csv"))

    # Create a mapping from neuron label to its index in the sorted list
    neuron_to_idx = {label: idx for idx, label in enumerate(NEURON_LABELS)}

    # Filter for only neurons in the list NEURON_LABELS
    neurons_all = set(NEURON_LABELS)
    df_master = df_master[df_master["label"].isin(neurons_all)]

    # Get positions for nodes from the master sheet
    pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
    pos = {
        neuron_to_idx[label]: [pos_dict[label]["x"], pos_dict[label]["y"], pos_dict[label]["z"]]
        for label in pos_dict
    }

    # Get neuron types from the master sheet
    df_master["type"] = df_master["type"].fillna("Unknown")  # Fill NaNs with 'Unknown'
    le = preprocessing.LabelEncoder()
    le.fit(df_master["type"].values)
    num_classes = len(le.classes_)
    y = torch.tensor(
        le.transform(df_master.set_index("label").reindex(NEURON_LABELS)["type"].values),
        dtype=torch.int32,
    )
    node_type = dict(zip(le.transform(le.classes_), le.classes_))

    # Get neuron classes from the master sheet
    df_master["class"] = df_master["class"].fillna("Unknown")  # Fill NaNs with 'Unknown'
    class_le = preprocessing.LabelEncoder()
    class_le.fit(df_master["class"].values)
    node_class = dict(zip(class_le.transform(class_le.classes_), class_le.classes_))

    # Create dummy feature matrix
    x = torch.empty(len(NEURON_LABELS), 1024, dtype=torch.float)  # filler data

    # Initialize node_label and n_id
    node_label = {idx: label for label, idx in neuron_to_idx.items()}
    n_id = torch.arange(len(NEURON_LABELS))

    # # Create the graph data object
    # graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y) # unused

    return x, y, num_classes, node_type, node_label, pos, n_id, node_class


def preprocess_default(raw_dir, save_as="graph_tensors.pt"):
    """
    Defaults to using a preprocessed version of the Cook et al. (2019) connectome data that
    was prepared by Kamal Premaratne (University of Miami) and extracted from the ZIP archive
    downloadable from https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip.
    """
    # Names of all C. elegans hermaphrodite neurons
    # NOTE: Only neurons in this list will be included in the connectomes constructed here.
    neurons_all = set(NEURON_LABELS)
    sep = r"[\t,]"

    # Chemical synapses nodes and edges
    GHermChem_Edges = pd.read_csv(os.path.join(raw_dir, "GHermChem_Edges.csv"), sep=sep)  # edges
    GHermChem_Nodes = pd.read_csv(os.path.join(raw_dir, "GHermChem_Nodes.csv"), sep=sep)  # nodes

    # Gap junctions
    GHermElec_Sym_Edges = pd.read_csv(
        os.path.join(raw_dir, "GHermElec_Sym_Edges.csv"), sep=sep
    )  # edges
    GHermElec_Sym_Nodes = pd.read_csv(
        os.path.join(raw_dir, "GHermElec_Sym_Nodes.csv"), sep=sep
    )  # nodes

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
    # NOTE: The first feature represents the weight of the gap junctions; the second feature represents the weight of the chemical synapses.
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

    # Load the neuron master sheet
    df_master = pd.read_csv(os.path.join(raw_dir, "neuron_master_sheet.csv"))
    df_master = df_master[df_master["label"].isin(neurons_all)]

    # Get neuron types from the master sheet
    df_master["type"] = df_master["type"].fillna("Unknown")  # Fill NaNs with 'Unknown'
    le = preprocessing.LabelEncoder()
    le.fit(df_master["type"].values)
    num_classes = len(le.classes_)
    y = torch.tensor(
        le.transform(df_master.set_index("label").reindex(NEURON_LABELS)["type"].values),
        dtype=torch.int32,
    )
    node_type = dict(zip(le.transform(le.classes_), le.classes_))

    # Get neuron classes from the master sheet
    df_master["class"] = df_master["class"].fillna("Unknown")  # Fill NaNs with 'Unknown'
    class_le = preprocessing.LabelEncoder()
    class_le.fit(df_master["class"].values)
    node_class = dict(zip(class_le.transform(class_le.classes_), class_le.classes_))

    # Create dummy feature matrix
    num_node_features = 1024
    x = torch.empty(len(NEURON_LABELS), num_node_features, dtype=torch.float)  # filler data

    # Get positions for nodes from the master sheet
    pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
    pos = {
        neuron_to_idx[label]: [pos_dict[label]["x"], pos_dict[label]["y"], pos_dict[label]["z"]]
        for label in pos_dict
    }

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
    neuron_to_idx = {label: idx for idx, label in enumerate(NEURON_LABELS)}
    pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
    pos = {
        neuron_to_idx[label]: [pos_dict[label]["x"], pos_dict[label]["y"], pos_dict[label]["z"]]
        for label in pos_dict
    }

    # Assign each node its global node index
    n_id = torch.arange(graph.num_nodes)
    node_label = {
        k: idx_to_neuron[k] for k in pos.keys()
    }  # names of neurons (e.g. AVAL, RIBR, VC1, etc.)

    # Save the tensors to use as raw data in the future.
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_randi_2023_funconn(raw_dir, save_as="graph_tensors_funconn.pt"):
    edges = []
    edge_attr = []

    # Load the Excel file
    xls = pd.ExcelFile(os.path.join(raw_dir, "CElegansFunctionalConnectivity.xlsx"))

    # Load the connectivity and significance matrices
    df_connectivity = pd.read_excel(xls, sheet_name=0, index_col=0)
    df_significance = pd.read_excel(xls, sheet_name=1, index_col=0)

    # Iterate over the connectivity matrix to keep only significant edges (q-value in significance matrix < 0.05)
    for i, (row_label, row) in enumerate(df_connectivity.iterrows()):
        for j, (col_label, value) in enumerate(row.items()):
            if pd.isna(value) or np.isnan(value):  # skip unmeasured edges
                continue
            if row_label in NEURON_LABELS and col_label in NEURON_LABELS:
                if df_significance.loc[row_label, col_label] < 0.05:
                    edges.append([row_label, col_label])
                    edge_attr.append([0, value])  # treat as if all edges are chemical synapses

    # NOTE: There is no second iteration over electrical synapse as there really are none.
    # We treated the functional weights as if they were chemical synapses for consistency with the other connectomes.

    # Create a mapping from neuron label to its index in the sorted list
    neuron_to_idx = {label: idx for idx, label in enumerate(NEURON_LABELS)}

    # Convert edge attributes and indices to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_openworm(raw_dir, save_as="graph_tensors_openworm.pt"):
    """
    Preprocess the OpenWorm connectome data.

    This function processes the raw OpenWorm connectome data into a format
    suitable for use in machine learning or graph analysis. It reads the
    raw data in .csv format, extracts relevant information, and creates
    graph tensors that represent the C. elegans connectome. The resulting
    graph tensors are saved in the 'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_openworm.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Load the OpenWorm connectome data
    df = pd.read_csv(os.path.join(raw_dir, "OpenWormConnectome.csv"), sep=r"[\t,]")

    edges = []
    edge_attr = []

    # Iterate through the dataframe to create edge lists and edge attributes
    for i in range(len(df)):
        neuron1 = df.loc[i, "Origin"]
        neuron2 = df.loc[i, "Target"]
        type = df.loc[i, "Type"]
        num_connections = df.loc[i, "Number of Connections"]

        if [neuron1, neuron2] not in edges:
            edges.append([neuron1, neuron2])
            if type == "GapJunction":
                edge_attr.append([num_connections, 0])
            else:
                edge_attr.append([0, num_connections])
        else:
            if type == "GapJunction":
                edge_attr[-1][0] = num_connections
            else:
                edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_witvliet_2020_7(raw_dir, save_as="graph_tensors_witvliet2020.pt"):
    """
    Preprocess the Witvliet et al. 2020 (adult 7) connectome data.

    This function processes the raw connectome data from Witvliet et al. 2020
    into a format suitable for use in machine learning or graph analysis. It
    reads the raw data in .csv format, extracts relevant information, and
    creates graph tensors that represent the C. elegans connectome. The
    resulting graph tensors are saved in the 'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_witvliet2020.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Load the Witvliet 2020 (adult 7) connectome data
    df = pd.read_csv(os.path.join(raw_dir, "witvliet_2020_7.csv"), sep=r"[\t,]")

    edges = []
    edge_attr = []

    # Iterate through the dataframe to create edge lists and edge attributes
    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]

        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]

            if [neuron1, neuron2] not in edges:
                edges.append([neuron1, neuron2])
                if type == "electrical":
                    edge_attr.append([num_connections, 0])
                else:
                    edge_attr.append([0, num_connections])
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_witvliet_2020_8(raw_dir, save_as="graph_tensors_witvliet2020.pt"):
    """
    Preprocess the Witvliet et al. 2020 (adult 8) connectome data.

    This function processes the raw connectome data from Witvliet et al. 2020
    into a format suitable for use in machine learning or graph analysis. It
    reads the raw data in .csv format, extracts relevant information, and
    creates graph tensors that represent the C. elegans connectome. The
    resulting graph tensors are saved in the 'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_witvliet2020.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Load the Witvliet 2020 (adult 8) connectome data
    df = pd.read_csv(os.path.join(raw_dir, "witvliet_2020_8.csv"), sep=r"[\t,]")

    edges = []
    edge_attr = []

    # Iterate through the dataframe to create edge lists and edge attributes
    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]

        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]

            if [neuron1, neuron2] not in edges:
                edges.append([neuron1, neuron2])
                if type == "electrical":
                    edge_attr.append([num_connections, 0])
                else:
                    edge_attr.append([0, num_connections])
            else:
                if type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_cook_2019(raw_dir, save_as="graph_tensors_cook2019.pt"):
    """
    Preprocess the Cook et al. 2019 connectome data.

    This function processes the raw connectome data from Cook et al. 2019 into a format
    suitable for use in machine learning or graph analysis. It reads the raw data in
    .xlsx format, extracts relevant information, and creates graph tensors that represent
    the C. elegans connectome. The resulting graph tensors are saved in the
    'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_cook2019.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    edges = []
    edge_attr = []

    # Chemical synapse processing
    df = pd.read_excel(os.path.join(raw_dir, "Cook2019.xlsx"), sheet_name="hermaphrodite chemical")

    for i, line in enumerate(df):
        if i > 2:
            col_data = df.iloc[:-1, i]
            for j, weight in enumerate(col_data):
                if j > 1 and not pd.isna(df.iloc[j, i]):
                    post = df.iloc[1, i]
                    pre = df.iloc[j, 2]
                    if pre in NEURON_LABELS and post in NEURON_LABELS:
                        edges.append([pre, post])
                        edge_attr.append([0, df.iloc[j, i]])

    # Gap junction processing
    df = pd.read_excel(
        os.path.join(raw_dir, "Cook2019.xlsx"), sheet_name="hermaphrodite gap jn asymmetric"
    )

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
                            edges.append([pre, post])
                            edge_attr.append([df.iloc[j, i], 0])

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_white_1986_whole(raw_dir, save_as="graph_tensors_white1986.pt"):
    """
    Preprocess the White et al. 1986 whole connectome data.

    This function processes the raw connectome data from White et al. 1986 into a format
    suitable for use in machine learning or graph analysis. It reads the raw data in
    .csv format, extracts relevant information, and creates graph tensors that represent
    the C. elegans connectome. The resulting graph tensors are saved in the
    'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_white1986.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_whole.csv"), sep=r"[\t,]")

    # Initialize lists to store edges and attributes
    origin = []
    target = []
    edges = []
    edge_attr = []

    # Process each row in the dataframe to extract edge information
    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]

        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin.append(neuron1)
            target.append(neuron2)

            synapse_type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]

            if [neuron1, neuron2] not in edges:
                edges.append([neuron1, neuron2])
                if synapse_type == "electrical":
                    edge_attr.append([num_connections, 0])
                else:
                    edge_attr.append([0, num_connections])
            else:
                if synapse_type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_white_1986_n2u(raw_dir, save_as="graph_tensors_white1986.pt"):
    """
    Preprocess the White et al. 1986 N2U connectome data.

    This function processes the raw connectome data from White et al. 1986 (N2U) into a format
    suitable for use in machine learning or graph analysis. It reads the raw data in
    .csv format, extracts relevant information, and creates graph tensors that represent
    the C. elegans connectome. The resulting graph tensors are saved in the
    'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_white1986.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_n2u.csv"), sep=r"[\t,]")

    # Initialize lists to store edges and attributes
    origin = []
    target = []
    edges = []
    edge_attr = []

    # Process each row in the dataframe to extract edge information
    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]

        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin.append(neuron1)
            target.append(neuron2)

            synapse_type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]

            if [neuron1, neuron2] not in edges:
                edges.append([neuron1, neuron2])
                if synapse_type == "electrical":
                    edge_attr.append([num_connections, 0])
                else:
                    edge_attr.append([0, num_connections])
            else:
                if synapse_type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_white_1986_jsh(raw_dir, save_as="graph_tensors_white1986.pt"):
    """
    Preprocess the White et al. 1986 JSH connectome data.

    This function processes the raw connectome data from White et al. 1986 (JSH) into a format
    suitable for use in machine learning or graph analysis. It reads the raw data in
    .csv format, extracts relevant information, and creates graph tensors that represent
    the C. elegans connectome. The resulting graph tensors are saved in the
    'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_white1986.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_jsh.csv"), sep=r"[\t,]")

    # Initialize lists to store edges and attributes
    origin = []
    target = []
    edges = []
    edge_attr = []

    # Process each row in the dataframe to extract edge information
    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]

        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin.append(neuron1)
            target.append(neuron2)

            synapse_type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]

            if [neuron1, neuron2] not in edges:
                edges.append([neuron1, neuron2])
                if synapse_type == "electrical":
                    edge_attr.append([num_connections, 0])
                else:
                    edge_attr.append([0, num_connections])
            else:
                if synapse_type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )


def preprocess_white_1986_jse(raw_dir, save_as="graph_tensors_white1986.pt"):
    """
    Preprocess the White et al. 1986 JSE connectome data.

    This function processes the raw connectome data from White et al. 1986 (JSE) into a format
    suitable for use in machine learning or graph analysis. It reads the raw data in
    .csv format, extracts relevant information, and creates graph tensors that represent
    the C. elegans connectome. The resulting graph tensors are saved in the
    'data/processed/connectome' folder.

    Parameters
    ----------
    raw_dir : str
        Directory with raw connectome data.
    save_as : str, optional
        Filename to save the processed graph tensors (default is 'graph_tensors_white1986.pt').

    Returns
    -------
    None
        This function does not return anything, but it saves the
        graph tensors in the 'data/processed/connectome' folder.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(raw_dir, "white_1986_jse.csv"), sep=r"[\t,]")

    # Initialize lists to store edges and attributes
    origin = []
    target = []
    edges = []
    edge_attr = []

    # Process each row in the dataframe to extract edge information
    for i in range(len(df)):
        neuron1 = df.loc[i, "pre"]
        neuron2 = df.loc[i, "post"]

        if neuron1 in NEURON_LABELS and neuron2 in NEURON_LABELS:
            origin.append(neuron1)
            target.append(neuron2)

            synapse_type = df.loc[i, "type"]
            num_connections = df.loc[i, "synapses"]

            if [neuron1, neuron2] not in edges:
                edges.append([neuron1, neuron2])
                if synapse_type == "electrical":
                    edge_attr.append([num_connections, 0])
                else:
                    edge_attr.append([0, num_connections])
            else:
                if synapse_type == "electrical":
                    edge_attr[-1][0] = num_connections
                else:
                    edge_attr[-1][-1] = num_connections

    # Create a mapping from neuron labels to their indices
    neuron_to_idx = dict(zip(NEURON_LABELS, range(len(NEURON_LABELS))))

    # Convert edge lists and edge attributes to tensors
    edge_attr = torch.tensor(edge_attr)
    edge_index = torch.tensor(
        [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
    ).T

    # Preprocess common tasks to get node features and other attributes
    x, y, num_classes, node_type, node_label, pos, n_id, node_class = preprocess_common_tasks(
        raw_dir, edge_index, edge_attr
    )

    # Create a dictionary to store graph tensors
    graph_tensors = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "num_classes": num_classes,
        "x": x,
        "y": y,
        "node_type": node_type,
        "node_label": node_label,
        "n_id": n_id,
        "node_class": node_class,
    }

    # Save the graph tensors to the specified file
    torch.save(
        graph_tensors,
        os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
    )
