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
    smooth_method="none",
    interpolate_method="linear",
    resample_dt=None,
    cleanup=False,
    **kwargs,
):
    """Preprocess and save C. elegans neural data to .pickle format.

    This function downloads and extracts the open-source datasets if not found in the
    root directory, preprocesses the neural data using the corresponding DatasetPreprocessor class,
    and then saves it to .pickle format. The processed data is saved in the
    data/processed/neural folder for further use.

    Args:
        url (str): Download link to a zip file containing the open-source data in raw form.
        zipfile (str): The name of the zipfile that is being downloaded.
        source_dataset (str, optional): The name of the source dataset to be pickled.
            If None or 'all', all datasets are pickled. Default is 'all'.
        transform (object, optional): The sklearn transformation to be applied to the data.
            Default is StandardScaler().
        smooth_method (str, optional): The smoothing method to apply to the data;
            options are 'gaussian', 'exponential', or 'moving'. Default is 'moving'.
        interpolate_method (str, optional): The scipy interpolation method to use when resampling the data.
            Default is 'linear'.
        resample_dt (float, optional): The resampling time interval in seconds.
            If None, no resampling is performed. Default is None.
        cleanup (bool, optional): If True, deletes the unzipped folder after processing. Default is False.
        **kwargs: Additional keyword arguments to be passed to the DatasetPreprocessor class.

    Returns:
        None

    Raises:
        AssertionError: If an invalid source dataset is requested.
        NameError: If the specified preprocessor class is not found.

    Steps:
        1. Construct paths for the zip file and source data.
        2. Create the neural data directory if it doesn't exist.
        3. Download and extract the zip file if the source data is not found.
        4. Instantiate and use the appropriate DatasetPreprocessor class to preprocess the data.
        5. Save the preprocessed data to .pickle format.
        6. Optionally, delete the unzipped folder if cleanup is True.
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
                # "{}/*".format(source_dataset),
                "-d",
                source_path,
                "-x",
                "__MACOSX/*",
            ]
            # Run the bash command
            std_out = subprocess.run(bash_command, text=True)
            # Output to log or terminal
            logger.info(f"Unzipping: {std_out} ...")
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


def get_presaved_datasets(url, file):
    """Download and unzip presaved data splits.

    This function downloads and extracts presaved data splits (commonly requested data patterns)
    from the specified URL. The extracted data is saved in the 'data' folder. The zip file is
    deleted after extraction.

    Args:
        url (str): The download link to the zip file containing the presaved data splits.
        file (str): The name of the zip file to be downloaded.

    Returns:
        None

    Steps:
        1. Construct the paths for the zip file and the data directory.
        2. Download the zip file from the specified URL.
        3. Extract the contents of the zip file to the data directory.
        4. Delete the zip file after extraction.
    """
    presaved_url = url
    presaved_file = file
    presave_path = os.path.join(ROOT_DIR, presaved_file)
    data_path = os.path.join(ROOT_DIR, "data")
    download_url(url=presaved_url, folder=ROOT_DIR, filename=presaved_file)
    extract_zip(presave_path, folder=data_path, delete_zip=True)
    return None


def preprocess_connectome(raw_files, pub=None):
    """Convert the raw connectome data to a graph tensor.

    This function processes raw connectome data, which includes chemical
    synapses and gap junctions, into a format suitable for use in machine
    learning or graph analysis. It reads the raw data in .csv format,
    processes it to extract relevant information, and creates graph
    tensors that represent the C. elegans connectome. The resulting
    graph tensors are saved in the 'data/processed/connectome' folder
    as 'graph_tensors.pt'. We distinguish between electrical (gap junction)
    and chemical synapses by using an edge attribute tensor with two feature dimensions:
    the first feature represents the weight of the gap junctions; and the second feature
    represents the weight of the chemical synapses.

    Args:
        raw_files (list): Contain the names of the raw connectome data to preprocess.
        pub (str, optional): The publication to use for preprocessing. Options include:
            - "openworm": OpenWorm project
            - "funconn" or "randi_2023": Randi et al., 2023 (functional connectivity)
            - "witvliet_7": Witvliet et al., 2020 (adult 7)
            - "witvliet_8": Witvliet et al., 2020 (adult 8)
            - "white_1986_whole": White et al., 1986 (whole)
            - "white_1986_n2u": White et al., 1986 (N2U)
            - "white_1986_jsh": White et al., 1986 (JSH)
            - "white_1986_jse": White et al., 1986 (JSE)
            - "cook_2019": Cook et al., 2019
            - None: Default to a preprocessed variant of Cook et al., 2019

    Returns:
        None

    Steps:
        1. Check that all necessary files are present.
        2. Download and extract the raw data if not found.
        3. Determine the appropriate preprocessing class based on the publication.
        4. Instantiate and use the appropriate preprocessor class to preprocess the data.
        5. Save the preprocessed graph tensors to a file.

    NOTE:
    * A connectome is a comprehensive map of the neural connections within
      an organism's brain or nervous system. It is essentially the wiring
      diagram of the brain, detailing how neurons and their synapses are
      interconnected.
    * The default connectome data used here is from Cook et al., 2019.
      If the raw data isn't found, please download the zip file from this link:
      https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip,
      unzip the archive in the data/raw folder, then run the MATLAB script `export_nodes_edges.m`.
    """
    # Check that all necessary files are present
    all_files_present = all([os.path.exists(os.path.join(RAW_DATA_DIR, rf)) for rf in raw_files])
    if not all_files_present:
        download_url(url=RAW_DATA_URL, folder=ROOT_DIR, filename=RAW_ZIP)
        extract_zip(
            path=os.path.join(ROOT_DIR, RAW_ZIP),
            folder=RAW_DATA_DIR,
            delete_zip=True,
        )

    # Determine appropriate preprocessing class based on publication
    preprocessors = {
        "openworm": OpenWormPreprocessor,
        "funconn": Randi2023Preprocessor,
        "randi_2023": Randi2023Preprocessor,
        "witvliet_7": Witvliet2020Preprocessor7,
        "witvliet_8": Witvliet2020Preprocessor8,
        "white_1986_whole": White1986WholePreprocessor,
        "white_1986_n2u": White1986N2UPreprocessor,
        "white_1986_jsh": White1986JSHPreprocessor,
        "white_1986_jse": White1986JSEPreprocessor,
        "cook_2019": Cook2019Preprocessor,
        None: DefaultPreprocessor,
    }

    preprocessor_class = preprocessors.get(pub, DefaultPreprocessor)
    preprocessor = preprocessor_class()
    preprocessor.preprocess()

    return None


class ConnectomeBasePreprocessor:
    """
    Base class for preprocessing connectome data.

    This class provides common methods and attributes for preprocessing connectome data,
    including loading neuron labels, loading a neuron master sheet, and performing common
    preprocessing tasks such as creating graph tensors and saving them.

    Attributes:
        neuron_labels (List[str]): List of neuron labels.
        neuron_master_sheet (pd.DataFrame): DataFrame containing the neuron master sheet.
        neuron_to_idx (dict): Dictionary mapping neuron labels to their corresponding indices.

    Methods:
        load_neuron_labels() -> List[str]:
            Loads the neuron labels from a file or a constant.
        load_neuron_master_sheet() -> pd.DataFrame:
            Loads the neuron master sheet from a CSV file.
        preprocess_common_tasks(edge_index, edge_attr):
            Performs common preprocessing tasks such as creating graph tensors.
        save_graph_tensors(save_as: str, graph, num_classes, node_type, node_label, n_id, node_class):
            Saves the graph tensors to a file.
    """

    def __init__(self):
        """Initializes the ConnectomeBasePreprocessor with neuron labels and master sheet.

        This constructor initializes the ConnectomeBasePreprocessor by loading the neuron labels
        and the neuron master sheet. It also creates a dictionary mapping neuron labels to their
        corresponding indices.

        Attributes:
            neuron_labels (List[str]): List of neuron labels.
            neuron_master_sheet (pd.DataFrame): DataFrame containing the neuron master sheet.
            neuron_to_idx (dict): Dictionary mapping neuron labels to their corresponding indices.
        """
        self.neuron_labels = self.load_neuron_labels()
        self.neuron_master_sheet = self.load_neuron_master_sheet()
        self.neuron_to_idx = {label: idx for idx, label in enumerate(self.neuron_labels)}

    def load_neuron_labels(self) -> List[str]:
        """Loads the neuron labels from a file or a constant.

        Returns:
            List[str]: A list of neuron labels.
        """
        return NEURON_LABELS

    def load_neuron_master_sheet(self) -> pd.DataFrame:
        """Loads the neuron master sheet from a CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the neuron master sheet.
        """
        return pd.read_csv(os.path.join(RAW_DATA_DIR, "neuron_master_sheet.csv"))

    def preprocess_common_tasks(self, edge_index, edge_attr):
        """Performs common preprocessing tasks such as creating graph tensors.

        This function processes the edge indices and attributes to create graph tensors
        that represent the connectome. It also ensures the symmetry of the gap junction
        adjacency matrix and adds missing nodes with zero-weight edges.

        Args:
            edge_index (torch.Tensor): Tensor containing the edge indices.
            edge_attr (torch.Tensor): Tensor containing the edge attributes.

        Returns:
            graph (torch_geometric.data.Data): The processed graph data object.
            num_classes (int): The number of unique neuron types.
            node_type (dict): Dictionary mapping neuron type indices to their labels.
            node_label (dict): Dictionary mapping node indices to neuron labels.
            n_id (torch.Tensor): Tensor containing the node indices.
            node_class (dict): Dictionary mapping neuron class indices to their labels.

        Steps:
            1. Filter the neuron master sheet to include only relevant neurons.
            2. Create a position dictionary for the neurons.
            3. Encode neuron types and classes using LabelEncoder.
            4. Initialize node features and labels.
            5. Add missing nodes with zero-weight edges to ensure a complete adjacency matrix.
            6. Ensure the symmetry of the gap junction adjacency matrix.
            7. Create the graph data object with the processed information.
        """
        df_master = self.neuron_master_sheet
        df_master = df_master[df_master["label"].isin(self.neuron_labels)]

        pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
        pos = {
            self.neuron_to_idx[label]: [
                pos_dict[label]["x"],
                pos_dict[label]["y"],
                pos_dict[label]["z"],
            ]
            for label in pos_dict
        }

        df_master["type"] = df_master["type"].fillna("Unknown")
        le = preprocessing.LabelEncoder()
        le.fit(df_master["type"].values)
        num_classes = len(le.classes_)
        y = torch.tensor(
            le.transform(df_master.set_index("label").reindex(self.neuron_labels)["type"].values),
            dtype=torch.int32,
        )
        node_type = dict(zip(le.transform(le.classes_), le.classes_))

        df_master["class"] = df_master["class"].fillna("Unknown")
        class_le = preprocessing.LabelEncoder()
        class_le.fit(df_master["class"].values)
        node_class = dict(zip(class_le.transform(class_le.classes_), class_le.classes_))

        x = torch.empty(len(self.neuron_labels), 1024, dtype=torch.float)

        node_label = {idx: label for label, idx in self.neuron_to_idx.items()}
        n_id = torch.arange(len(self.neuron_labels))

        # Add missing nodes with zero-weight edges to ensure the adjacency matrix is 300x300
        all_indices = torch.arange(len(self.neuron_labels))
        full_edge_index = torch.combinations(all_indices, r=2).T
        existing_edges_set = set(map(tuple, edge_index.T.tolist()))

        additional_edges = []
        additional_edge_attr = []
        for edge in full_edge_index.T.tolist():
            if tuple(edge) not in existing_edges_set:
                additional_edges.append(edge)
                additional_edge_attr.append([0, 0])

        if additional_edges:
            additional_edges = torch.tensor(additional_edges).T
            additional_edge_attr = torch.tensor(additional_edge_attr, dtype=torch.float)
            edge_index = torch.cat([edge_index, additional_edges], dim=1)
            edge_attr = torch.cat([edge_attr, additional_edge_attr], dim=0)

        # Check for symmetry in the gap junction adjacency matrix
        gap_junctions = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr[:, 0]).squeeze(0)
        if not torch.allclose(gap_junctions.T, gap_junctions):
            raise AssertionError("The gap junction adjacency matrix is not symmetric.")

        # Create the graph data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.pos = pos  # Add positions to the graph object

        return graph, num_classes, node_type, node_label, n_id, node_class

    def save_graph_tensors(
        self,
        save_as: str,
        graph,
        num_classes,
        node_type,
        node_label,
        n_id,
        node_class,
    ):
        """Saves the graph tensors to a file.

        Args:
            save_as (str): The name of the file to save the graph tensors to.
            graph (torch_geometric.data.Data): The processed graph data object.
            num_classes (int): The number of unique neuron types.
            node_type (dict): Dictionary mapping neuron type indices to their labels.
            node_label (dict): Dictionary mapping node indices to neuron labels.
            n_id (torch.Tensor): Tensor containing the node indices.
            node_class (dict): Dictionary mapping neuron class indices to their labels.
        """
        graph_tensors = {
            "edge_index": graph.edge_index,
            "edge_attr": graph.edge_attr,
            "pos": graph.pos,
            "num_classes": num_classes,
            "x": graph.x,
            "y": graph.y,
            "node_type": node_type,
            "node_label": node_label,
            "n_id": n_id,
            "node_class": node_class,
        }

        torch.save(
            graph_tensors,
            os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
        )


class DefaultPreprocessor(ConnectomeBasePreprocessor):
    """
    Default preprocessor for connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the default connectome data. It includes methods for loading, processing,
    and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors.pt"):
            Preprocesses the connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors.pt"):
        """
        Preprocesses the connectome data and saves the graph tensors to a file.

        The data is read from multiple CSV files named "GHermChem_Edges.csv",
        "GHermChem_Nodes.csv", "GHermGap_Edges.csv", and "GHermGap_Nodes.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors.pt".

        Steps:
            1. Load the chemical synapse edges and nodes from "GHermChem_Edges.csv" and "GHermChem_Nodes.csv".
            2. Load the electrical synapse edges and nodes from "GHermGap_Edges.csv" and "GHermGap_Nodes.csv".
            3. Initialize sets for all C. elegans hermaphrodite neurons.
            4. Process the chemical synapse edges and nodes:
                - Filter edges and nodes based on neuron labels.
                - Append edges and attributes to the respective lists.
            5. Process the electrical synapse edges and nodes:
                - Filter edges and nodes based on neuron labels.
                - Append edges and attributes to the respective lists.
            6. Convert edge attributes and edge indices to tensors.
            7. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            8. Save the graph tensors to the specified file.
        """
        # >>>>>>>>
        # Hack to override DefaultProprecessor with Witvliet2020Preprocessor7 which is a more up-to-date connectome of C. elegans.
        return Witvliet2020Preprocessor7.preprocess(self, save_as="graph_tensors.pt")
        # <<<<<<<
        # Names of all C. elegans hermaphrodite neurons
        neurons_all = set(self.neuron_labels)
        sep = r"[\t,]"

        # Chemical synapses nodes and edges
        GHermChem_Edges = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermChem_Edges.csv"), sep=sep
        )  # edges
        GHermChem_Nodes = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermChem_Nodes.csv"), sep=sep
        )  # nodes

        # Gap junctions
        GHermElec_Sym_Edges = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermElec_Sym_Edges.csv"), sep=sep
        )  # edges
        GHermElec_Sym_Nodes = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermElec_Sym_Nodes.csv"), sep=sep
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
        neuron_to_idx = dict(zip(self.neuron_labels, range(len(self.neuron_labels))))

        # edge_index for gap junctions
        arr = Ggap_edges[["EndNodes_1", "EndNodes_2"]].values
        ggap_edge_index = torch.empty(*arr.shape, dtype=torch.long)
        for i, row in enumerate(arr):
            ggap_edge_index[i, :] = torch.tensor([neuron_to_idx[x] for x in row], dtype=torch.long)
        ggap_edge_index = ggap_edge_index.T  # [2, num_edges]
        # Do the reverse direction to ensure symmetry of gap junctions
        ggap_edge_index = torch.hstack((ggap_edge_index, ggap_edge_index[[1, 0], :]))

        # edge_index for chemical synapses
        arr = Gsyn_edges[["EndNodes_1", "EndNodes_2"]].values
        gsyn_edge_index = torch.empty(*arr.shape, dtype=torch.long)
        for i, row in enumerate(arr):
            gsyn_edge_index[i, :] = torch.tensor([neuron_to_idx[x] for x in row], dtype=torch.long)
        gsyn_edge_index = gsyn_edge_index.T  # [2, num_edges]

        # edge attributes
        # NOTE: The first feature represents the weight of the gap junctions;
        # The second feature represents the weight of the chemical synapses.
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
        # Do the reverse direction to ensure symmetry of gap junctions
        ggap_edge_attr = torch.vstack((ggap_edge_attr, ggap_edge_attr))

        # edge_attr for chemical synapses
        num_edges = len(Gsyn_edges)
        gsyn_edge_attr = torch.empty(
            num_edges, num_edge_features, dtype=torch.float
        )  # [num_edges, num_edge_features]
        for i, weight in enumerate(Gsyn_edges.Weight.values):
            gsyn_edge_attr[i, :] = torch.tensor(
                [0, weight], dtype=torch.float
            )  # chemical synapse encoded as [0,1]

        # Merge electrical and chemical graphs into a single connectome graph
        combined_edge_index = torch.hstack((ggap_edge_index, gsyn_edge_index))
        combined_edge_attr = torch.vstack((ggap_edge_attr, gsyn_edge_attr))
        edge_index, edge_attr = coalesce(
            combined_edge_index, combined_edge_attr, reduce="add"
        )  # features = [elec_wt, chem_wt]

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class OpenWormPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the OpenWorm connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the OpenWorm connectome data. It includes methods for loading, processing,
    and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_openworm.pt"):
            Preprocesses the OpenWorm connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_openworm.pt"):
        """
        Preprocesses the OpenWorm connectome data and saves the graph tensors to a file.

        The data is read from a CSV file named "OpenWormConnectome.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_openworm.pt".

        Steps:
            1. Load the connectome data from "OpenWormConnectome.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "OpenWormConnectome.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "Origin"]
            neuron2 = df.loc[i, "Target"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                synapse_type = df.loc[i, "Type"]
                num_connections = df.loc[i, "Number of Connections"]
                edges.append([neuron1, neuron2])
                if synapse_type == "GapJunction":  # electrical synapse
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif synapse_type == "Send":  # chemical synapse
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class Randi2023Preprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Randi et al., 2023 connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Randi et al., 2023 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_funconn.pt"):
            Preprocesses the Randi et al., 2023 connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_funconn.pt"):
        """
        Preprocesses the Randi et al., 2023 connectome data and saves the graph tensors to a file.

        The data is read from an Excel file named "CElegansFunctionalConnectivity.xlsx".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_funconn.pt".

        Steps:
            1. Load the connectivity and significance data from "CElegansFunctionalConnectivity.xlsx".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the connectivity DataFrame:
                - Extract neuron pairs and their connectivity values.
                - Check significance and append edges and attributes to the respective lists.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        edges = []
        edge_attr = []

        xls = pd.ExcelFile(os.path.join(RAW_DATA_DIR, "CElegansFunctionalConnectivity.xlsx"))
        df_connectivity = pd.read_excel(xls, sheet_name=0, index_col=0)
        df_significance = pd.read_excel(xls, sheet_name=1, index_col=0)

        for i, (row_label, row) in enumerate(df_connectivity.iterrows()):
            for j, (col_label, value) in enumerate(row.items()):
                if pd.isna(value) or np.isnan(value):
                    continue
                if row_label in self.neuron_labels and col_label in self.neuron_labels:
                    if df_significance.loc[row_label, col_label] < 0.05:
                        edges.append([row_label, col_label])
                        edge_attr.append([0, value])

        neuron_to_idx = {label: idx for idx, label in enumerate(self.neuron_labels)}
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class Witvliet2020Preprocessor7(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Witvliet et al., 2020 connectome data (adult 7).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Witvliet et al., 2020 connectome data for adult 7. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_witvliet2020_7.pt"):
            Preprocesses the Witvliet et al., 2020 connectome data for adult 7 and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_witvliet2020_7.pt"):
        """
        Preprocesses the Witvliet et al., 2020 connectome data for adult 7 and saves the graph tensors to a file.

        The data is read from a CSV file named "witvliet_2020_7.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_witvliet2020_7.pt".

        Steps:
            1. Load the connectome data from "witvliet_2020_7.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "witvliet_2020_7.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class Witvliet2020Preprocessor8(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Witvliet et al., 2020 connectome data (adult 8).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Witvliet et al., 2020 connectome data for adult 8. It includes methods
    for loading, processing, and saving the connectome data.

    Methods
    -------
    preprocess(save_as="graph_tensors_witvliet2020_8.pt")
        Preprocesses the Witvliet et al., 2020 connectome data for adult 8 and saves the graph tensors to a file.
        The data is read from a CSV file named "witvliet_2020_8.csv".
    """

    def preprocess(self, save_as="graph_tensors_witvliet2020_8.pt"):
        """
        Preprocesses the Witvliet et al., 2020 connectome data for adult 8 and saves the graph tensors to a file.

        The data is read from a CSV file named "witvliet_2020_8.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_witvliet2020_8.pt".

        Steps:
            1. Load the connectome data from "witvliet_2020_8.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "witvliet_2020_8.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class Cook2019Preprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Cook et al., 2019 connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Cook et al., 2019 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_cook2019.pt"):
            Preprocesses the Cook et al., 2019 connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_cook2019.pt"):
        """
        Preprocesses the Cook et al., 2019 connectome data and saves the graph tensors to a file.

        The data is read from an Excel file named "Cook2019.xlsx".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_cook2019.pt".

        Steps:
            1. Load the chemical synapse data from the "hermaphrodite chemical" sheet in "Cook2019.xlsx".
            2. Load the electrical synapse data from the "hermaphrodite gap jn symmetric" sheet in "Cook2019.xlsx".
            3. Initialize lists for edges and edge attributes.
            4. Iterate through the chemical synapse data:
                - Extract neuron pairs and their weights.
                - Append edges and attributes to the respective lists.
            5. Iterate through the electrical synapse data:
                - Extract neuron pairs and their weights.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            6. Convert edge attributes and edge indices to tensors.
            7. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            8. Save the graph tensors to the specified file.
        """
        edges = []
        edge_attr = []

        xlsx_file = pd.ExcelFile(os.path.join(RAW_DATA_DIR, "Cook2019.xlsx"))

        df = pd.read_excel(xlsx_file, sheet_name="hermaphrodite chemical")

        for i, line in enumerate(df):
            if i > 2:
                col_data = df.iloc[:-1, i]
                for j, weight in enumerate(col_data):
                    if j > 1 and not pd.isna(df.iloc[j, i]):
                        post = df.iloc[1, i]
                        pre = df.iloc[j, 2]
                        if pre in self.neuron_labels and post in self.neuron_labels:
                            edges.append([pre, post])
                            edge_attr.append(
                                [0, df.iloc[j, i]]
                            )  # second edge_attr feature is for gap junction weights

        df = pd.read_excel(xlsx_file, sheet_name="hermaphrodite gap jn symmetric")

        for i, line in enumerate(df):
            if i > 2:
                col_data = df.iloc[:-1, i]
                for j, weight in enumerate(col_data):
                    if j > 1 and not pd.isna(df.iloc[j, i]):
                        post = df.iloc[1, i]
                        pre = df.iloc[j, 2]
                        if pre in self.neuron_labels and post in self.neuron_labels:
                            if [pre, post] in edges:
                                edge_idx = edges.index([pre, post])
                                edge_attr[edge_idx][0] = df.iloc[
                                    j, i
                                ]  # first edge_attr feature is for gap junction weights
                            else:
                                edges.append([pre, post])
                                edge_attr.append([df.iloc[j, i], 0])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class White1986WholePreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (whole).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the whole organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_whole.pt"):
            Preprocesses the White et al., 1986 connectome data for the whole organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_whole.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the whole organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_whole.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_whole.pt".

        Steps:
            1. Load the connectome data from "white_1986_whole.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_whole.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class White1986N2UPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (N2U).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the N2U organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_n2u.pt"):
            Preprocesses the White et al., 1986 connectome data for the N2U organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_n2u.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the N2U organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_n2u.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_n2u.pt".

        Steps:
            1. Load the connectome data from "white_1986_n2u.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_n2u.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class White1986JSHPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (JSH).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the JSH organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_jsh.pt"):
            Preprocesses the White et al., 1986 connectome data for the JSH organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_jsh.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the JSH organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_jsh.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_jsh.pt".

        Steps:
            1. Load the connectome data from "white_1986_jsh.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_jsh.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


class White1986JSEPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (JSE).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the JSE organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_jse.pt"):
            Preprocesses the White et al., 1986 connectome data for the JSE organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_jse.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the JSE organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_jse.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_jse.pt".

        Steps:
            1. Load the connectome data from "white_1986_jse.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_jse.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        graph, num_classes, node_type, node_label, n_id, node_class = self.preprocess_common_tasks(
            edge_index, edge_attr
        )

        self.save_graph_tensors(
            save_as,
            graph,
            num_classes,
            node_type,
            node_label,
            n_id,
            node_class,
        )


def extract_zip(path: str, folder: str = None, log: bool = True, delete_zip: bool = True):
    """Extracts a zip archive to a specific folder while ignoring the __MACOSX directory.

    Args:
        path (str): The path to the zip archive.
        folder (str, optional): The folder where the files will be extracted to. Defaults to the parent of `path`.
        log (bool, optional): If False, will not print anything to the console. Default is True.
        delete_zip (bool, optional): If True, will delete the zip archive after extraction. Default is True.

    Steps:
        1. Determine the extraction folder. If not provided, use the parent directory of the zip file.
        2. Log the extraction process if logging is enabled.
        3. Open the zip file and iterate through its members.
            - Skip any members that are part of the __MACOSX directory.
            - Extract the remaining members to the specified folder.
        4. Delete the zip file if `delete_zip` is True.
    """
    print(path)

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


###############################################################################################
# TODO: Encapsulate smoothing functions in OOP style class.
def gaussian_kernel_smooth(x, t, sigma):
    """Causal Gaussian smoothing for a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        x_smooth (ndarray): The smoothed time series.

    Steps:
        1. Check if the input is a torch.Tensor and convert to numpy if necessary.
        2. Reshape the input if it is 1-dimensional.
        3. Initialize the smoothed time series array.
        4. Compute the Gaussian weights for each time point.
        5. Apply the Gaussian smoothing to each time point and feature.
        6. Convert the smoothed time series back to torch.Tensor if the input was a tensor.
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
    """Causal moving average smoothing filter for a multidimensional time series.

    Parameters:
        x (ndarray): The input time series to be smoothed.
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        window_size (int): The size of the moving average window. Must be an odd number.

    Returns:
        x_smooth (ndarray): The smoothed time series.

    Steps:
        1. Ensure window_size is odd for symmetry.
        2. Check for correct dimensions and convert to torch.Tensor if necessary.
        3. Initialize the smoothed time series array.
        4. Apply the moving average smoothing to each time point and feature.
        5. Convert the smoothed time series back to numpy.ndarray if the input was a numpy array.
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
    This method is already causal by its definition.

    Parameters:
        x (ndarray): The input time series to be smoothed (time, neurons).
        t (ndarray): The time vector (in seconds) corresponding to the input time series.
        alpha (float): The smoothing factor, 0 < alpha < 1. A higher value of alpha will
                       result in less smoothing (more weight is given to the current value),
                       while a lower value of alpha will result in more smoothing
                       (more weight is given to the previous smoothed values).

    Returns:
        x_smooth (ndarray): The smoothed time series.

    Steps:
        1. Check if the input is a torch.Tensor and convert to numpy if necessary.
        2. Reshape the input if it is 1-dimensional.
        3. Initialize the smoothed time series array.
        4. Apply the exponential smoothing to each time point and feature.
        5. Convert the smoothed time series back to torch.Tensor if the input was a tensor.
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

    Parameters:
        calcium_data (np.ndarray): Original calcium data with shape (time, neurons).
        time_in_seconds (np.ndarray): Time vector with shape (time, 1).
        smooth_method (str): The method used to smooth the data. Options are "gaussian", "moving", "exponential".

    Returns:
        smooth_ca_data (np.ndarray): Calcium data that is smoothed.

    Steps:
        1. Check if the smooth_method is None, and if so, return the original calcium data.
        2. If the smooth_method is "gaussian", apply Gaussian kernel smoothing.
        3. If the smooth_method is "moving", apply moving average smoothing.
        4. If the smooth_method is "exponential", apply exponential kernel smoothing.
        5. Raise a TypeError if the smooth_method is not recognized.
    """
    if smooth_method is None:
        smooth_ca_data = calcium_data
    elif str(smooth_method).lower() == "gaussian":
        smooth_ca_data = gaussian_kernel_smooth(
            calcium_data, time_in_seconds, sigma=kwargs.get("sigma", 5)
        )
    elif str(smooth_method).lower() == "moving":
        smooth_ca_data = moving_average_smooth(
            calcium_data, time_in_seconds, window_size=kwargs.get("window_size", 15)
        )
    elif str(smooth_method).lower() == "exponential":
        smooth_ca_data = exponential_kernel_smooth(
            calcium_data, time_in_seconds, alpha=kwargs.get("alpha", 0.5)
        )
    elif str(smooth_method).lower() == "none":
        smooth_ca_data = calcium_data
    else:
        raise TypeError("See `configs/submodule/preprocess.yaml` for viable smooth methods.")
    return smooth_ca_data


###############################################################################################


def reshape_calcium_data(worm_dataset):
    """Reorganizes calcium data into a standard organized matrix with shape (max_timesteps, NUM_NEURONS).
    Also creates neuron masks and mappings of neuron labels to indices in the data.
    Converts the data to torch tensors.

    Parameters:
        worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.

    Returns:
        dict: The modified worm dataset with restructured calcium data.

    Steps:
        1. Initialize the CalciumDataReshaper with the provided worm dataset.
        2. Return the reshaped worm dataset.
    """
    reshaper = CalciumDataReshaper(worm_dataset)
    return reshaper.worm_dataset


def interpolate_data(time, data, target_dt, method="linear"):
    """
    Interpolate data using np.interp.

    This function takes the given time points and corresponding data and
    interpolates them to create new data points with the desired time interval.

    Parameters:
        time (numpy.ndarray): 1D array containing the time points corresponding to the data.
        data (numpy.ndarray): A 2D array containing the data to be interpolated, with shape (time, neurons).
        target_dt (float): The desired time interval between the interpolated data points. If None, no interpolation is performed.
        method (str, optional): The scipy interpolation method to use when resampling the data. Default is 'linear'.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the interpolated time points and data.

    Steps:
        1. Check if correct interpolation method is provided.
        2. If target_dt is None, return the original data.
        3. Ensure that time is a 1D array.
        4. Interpolate the data.
        5. Reshape interpolated time vector to (time, 1).
        6. Return the interpolated data.
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
    """
    Downsample data using aggregation.

    This function takes the given time points and corresponding data and
    downsamples them by averaging over intervals defined by `target_dt`.

    Parameters:
        time (numpy.ndarray): 1D array containing the time points corresponding to the data,
                              with shape (time, 1).
        data (numpy.ndarray): A 2D array containing the data to be downsampled, with shape
                              (time, neurons).
        target_dt (float): The desired time interval between the downsampled data points.
                           If None, no downsampling is performed.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the downsampled time points and data.

    Steps:
        1. If target_dt is None, return the original data.
        2. Ensure that time is a 1D array.
        3. Compute the downsample rate.
        4. Determine the number of intervals.
        5. Create the downsampled time array.
        6. Downsample the data by averaging over intervals.
        7. Reshape downsampled time vector to (time, 1).
        8. Return the downsampled data.
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
    """
    A transform for causal normalization of time series data.

    This normalizer computes the mean and standard deviation up to each time point t,
    ensuring that the normalization at each time point is based solely on past
    and present data, maintaining the causal nature of the time series.

    Attributes:
        nan_fill_method (str): Method to fill NaN values. Options are 'ffill' (forward fill),
                               'bfill' (backward fill), and 'interpolate'. Default is 'interpolate'.
        cumulative_mean_ (np.ndarray): Cumulative mean up to each time point.
        cumulative_std_ (np.ndarray): Cumulative standard deviation up to each time point.

    Methods:
        fit(X, y=None):
            Compute the cumulative mean and standard deviation of the dataset X.
        transform(X):
            Perform causal normalization on the dataset X using the previously computed cumulative mean and standard deviation.
        fit_transform(X, y=None):
            Fit to data, then transform it.
    """

    def __init__(self, nan_fill_method="interpolate"):
        """
        Initialize the CausalNormalizer with a method to handle NaN values.

        Parameters:
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
            X (array-like): The input data with potential NaN values.

        Returns:
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

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data.
            y (Ignored): Not used, present here for API consistency by convention.

        Returns:
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
        """
        Perform causal normalization on the dataset X using the
        previously computed cumulative mean and standard deviation.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data.

        Returns:
            X_transformed (array-like of shape (n_samples, n_features)): The transformed data.
        """
        if self.cumulative_mean_ is None or self.cumulative_std_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        X_transformed = (X - self.cumulative_mean_) / self.cumulative_std_
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit and transform.
            y (Ignored): Not used, present here for API consistency by convention.

        Returns:
            X_transformed (array-like of shape (n_samples, n_features)): The transformed data.
        """
        return self.fit(X).transform(X)


class CalciumDataReshaper:
    """
    Reshapes and organizes calcium imaging data for a single worm.

    This class takes a dataset for a single worm and reorganizes the calcium data into a standard
    matrix with shape (max_timesteps, NUM_NEURONS). It also creates neuron masks and mappings of
    neuron labels to indices in the data, and converts the data to torch tensors.

    Attributes:
        worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.
        named_neuron_to_idx (dict): Mapping of named neurons to their indices.
        unknown_neuron_to_idx (dict): Mapping of unknown neurons to their indices.
        slot_to_named_neuron (dict): Mapping of slots to named neurons.
        slot_to_unknown_neuron (dict): Mapping of slots to unknown neurons.
        slot_to_neuron (dict): Mapping of slots to neurons.
        dtype (torch.dtype): Data type for the tensors.

    Methods:
        _init_neuron_data():
            Initializes attributes from keys that must already be present in the worm dataset.
        _reshape_data():
            Reshapes the calcium data and updates the worm dataset.
        _prepare_initial_data():
            Prepares initial data structures for reshaping.
        _init_empty_calcium_data():
            Initializes empty calcium data matrices.
        _tensor_time_data():
            Converts time data to torch tensors.
        _fill_named_neurons_data():
            Fills data for named neurons.
        _fill_calcium_data(idx, slot):
            Fills calcium data for a given neuron index and slot.
        _fill_unknown_neurons_data():
            Fills data for unknown neurons.
        _update_worm_dataset():
            Updates the worm dataset with reshaped data and mappings.
        _remove_old_mappings():
            Removes old mappings from the worm dataset.
    """

    def __init__(self, worm_dataset: dict):
        """
        Initialize the CalciumDataReshaper with the provided worm dataset.

        Parameters:
            worm_dataset (dict): Dataset for a single worm that includes calcium data and other information.

        NOTE:
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
        """
        Reshapes the calcium data and updates the worm dataset.
        """
        self._prepare_initial_data()
        self._fill_named_neurons_data()
        self._fill_unknown_neurons_data()
        self._update_worm_dataset()
        self._remove_old_mappings()

    def _prepare_initial_data(self):
        """
        Prepares initial data structures for reshaping.
        """
        assert (
            len(self.idx_to_neuron) == self.calcium_data.shape[1]
        ), "Number of neurons in calcium data matrix does not match number of recorded neurons."
        self.named_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self.unknown_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self._init_empty_calcium_data()
        self._tensor_time_data()

    def _init_empty_calcium_data(self):
        """
        Initializes empty calcium data matrices.
        """
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
        """
        Converts time data to torch tensors.
        """
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
        """
        Fills data for named neurons.
        """
        for slot, neuron in enumerate(NEURON_LABELS):
            if neuron in self.neuron_to_idx:  # named neuron
                idx = self.neuron_to_idx[neuron]
                self.named_neuron_to_idx[neuron] = idx
                self._fill_calcium_data(idx, slot)
                self.named_neurons_mask[slot] = True
                self.slot_to_named_neuron[slot] = neuron

    def _fill_calcium_data(self, idx, slot):
        """
        Fills calcium data for a given neuron index and slot.

        Parameters:
            idx (int): Index of the neuron in the original dataset.
            slot (int): Slot in the reshaped dataset.
        """
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
        """
        Fills data for unknown neurons.
        """
        free_slots = list(np.where(~self.named_neurons_mask)[0])
        for neuron in set(self.neuron_to_idx) - set(self.named_neuron_to_idx):
            self.unknown_neuron_to_idx[neuron] = self.neuron_to_idx[neuron]
            slot = np.random.choice(free_slots)
            free_slots.remove(slot)
            self.slot_to_unknown_neuron[slot] = neuron
            self._fill_calcium_data(self.neuron_to_idx[neuron], slot)
            self.unknown_neurons_mask[slot] = True

    def _update_worm_dataset(self):
        """
        Updates the worm dataset with reshaped data and mappings.
        """
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
        """
        Removes old mappings from the worm dataset.
        """
        keys_to_delete = [key for key in self.worm_dataset if "idx" in key]
        for key in keys_to_delete:
            self.worm_dataset.pop(key, None)


class NeuralBasePreprocessor:
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
        class SpecificDatasetPreprocessor(NeuralBasePreprocessor):
            def load_data(self):
                # Implement dataset-specific loading logic here.
    """

    def __init__(
        self,
        dataset_name,
        # TODO: Try different transforms from sklearn such as QuantileTransformer, etc. as well as custom CausalNormalizer.
        transform=StandardScaler(),  # StandardScaler() #PowerTransformer() #CausalNormalizer() #None
        smooth_method="none",
        interpolate_method="linear",
        resample_dt=0.1,
        **kwargs,
    ):
        """
        Initialize the NeuralBasePreprocessor with the provided parameters.

        Parameters:
            dataset_name (str): The name of the dataset to be preprocessed.
            transform (object, optional): The sklearn transformation to be applied to the data. Default is StandardScaler().
            smooth_method (str, optional): The smoothing method to apply to the data. Default is "moving".
            interpolate_method (str, optional): The interpolation method to use when resampling the data. Default is "linear".
            resample_dt (float, optional): The resampling time interval in seconds. Default is 0.1.
            **kwargs: Additional keyword arguments for smoothing.
        """
        self.source_dataset = dataset_name
        self.transform = transform
        self.smooth_method = smooth_method
        self.interpolate_method = interpolate_method
        self.resample_dt = resample_dt
        self.smooth_kwargs = kwargs
        self.raw_data_path = os.path.join(ROOT_DIR, "opensource_data")
        self.processed_data_path = os.path.join(ROOT_DIR, "data/processed/neural")

    def smooth_data(self, data, time_in_seconds):
        """
        Smooth the data using the specified smoothing method.

        Parameters:
            data (np.ndarray): The input data to be smoothed.
            time_in_seconds (np.ndarray): The time vector corresponding to the input data.

        Returns:
            np.ndarray: The smoothed data.
        """
        return smooth_data_preprocess(
            data,
            time_in_seconds,
            self.smooth_method,
            **self.smooth_kwargs,
        )

    def resample_data(self, time_in_seconds, data, upsample=True):
        """
        Resample the data to the desired time interval.

        Parameters:
            time_in_seconds (np.ndarray): Time vector in seconds with shape (time, 1).
            data (np.ndarray): Original, non-uniformly sampled calcium data with shape (time, neurons).
            upsample (bool, optional): Whether to sample at a higher frequency (i.e., with smaller dt). Default is True.

        Returns:
            np.ndarray, np.ndarray: Resampled time vector and calcium data.
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
        """
        Normalize the data using the specified transformation.

        Parameters:
            data (np.ndarray): The input data to be normalized.

        Returns:
            np.ndarray: The normalized data.
        """
        if self.transform is None:
            return data
        return self.transform.fit_transform(data)

    def save_data(self, data_dict):
        """
        Save the processed data to a .pickle file.

        Parameters:
            data_dict (dict): The processed data to be saved.
        """
        file = os.path.join(self.processed_data_path, f"{self.source_dataset}.pickle")
        with open(file, "wb") as f:
            pickle.dump(data_dict, f)

    def create_neuron_idx(self, unique_IDs):
        """
        Create a neuron label to index mapping from the raw data.

        Parameters:
            unique_IDs (list): List of unique neuron IDs.

        Returns:
            dict: Mapping of neuron labels to indices.
            int: Number of named neurons.
        """
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
        Load the raw data from a .mat file.

        Parameters:
            file_name (str): The name of the file to load.

        Returns:
            dict: The loaded data.
        """
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))

    def extract_data(self):
        """
        Extract the basic data from the raw data file.

        This method should be overridden by subclasses to implement dataset-specific extraction logic.
        """
        raise NotImplementedError()

    def create_metadata(self):
        """
        Create a dictionary of extra information or metadata for a dataset.

        Returns:
            dict: A dictionary of extra information or metadata.
        """
        extra_info = dict()
        return extra_info

    def pick_non_none(self, l):
        """
        Returns the first non-None element in a list.

        Parameters:
            l (list): The input list.

        Returns:
            The first non-None element in the list.
        """
        for i in range(len(l)):
            if l[i] is not None:
                return l[i]
        return None

    def preprocess(self):
        """
        Main preprocessing method that calls the other methods in the class.

        This method should be overridden by subclasses to implement dataset-specific preprocessing logic.
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

        Parameters:
            neuron_IDs (list): List of arrays of neuron IDs.
            traces (list): List of arrays of calcium traces, with indices corresponding to neuron_IDs.
            raw_timeVectorSeconds (list): List of arrays of time vectors, with indices corresponding to neuron_IDs.
            preprocessed_data (dict): Dictionary of preprocessed data from previous worms that gets extended with more worms here.
            worm_idx (int): Index of the current worm.

        Returns:
            dict: Collection of all preprocessed worm data so far.
            int: Index of the next worm to preprocess.

        Steps:
            1. Iterate through the traces and preprocess each one.
            2. Normalize the calcium data.
            3. Compute calcium dynamics (residual calcium).
            4. Smooth the data.
            5. Resample the data (raw and smoothed data).
            6. Name the worm and update the index.
            7. Save the data.
        """

        for i, trace_data in enumerate(traces):
            # Matrix `trace_data` should be shaped as (time, neurons)
            assert trace_data.ndim == 2, "Calcium traces must be 2D arrays."
            assert trace_data.shape[1] == len(
                neuron_IDs[i]
            ), "Calcium trace does not have the right number of neurons."
            # Ignore any worms with empty traces
            if trace_data.size == 0:
                continue
            # Map named neurons
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
            ## DEBUG ###
            # Skip worms with no labelled neurons.
            # TODO: Should we do this? What if we want to infer these using trained models as a dowstream task?
            if num_named_neurons == 0:
                continue

            logger.info(f"DEBUG. trace_data.shape = {trace_data.shape}") # DEBUG
            ## DEBUG ###
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

            logger.info(f"DEBUG. calcium_data.shape = {calcium_data.shape}") # DEBUG
            logger.info(f"DEBUG. time_in_seconds.shape = {time_in_seconds.shape}") # DEBUG

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
            ), f"Resampling failed. The median dt ({resampled_median_dt}) of the resampled time vector is different from desired dt ({self.resample_dt})."
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


class Nejatbakhsh2020Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Nejatbakhsh et al., 2020 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Nejatbakhsh et al., 2020 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(file): Extracts neuron IDs, calcium traces, and time vector from the NWB file.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Nejatbakhsh et al., 2020 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Nejatbakhsh2020Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Nejatbakhsh2020",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the NWB file.

        Parameters:
            file (str): The path to the NWB file.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        with NWBHDF5IO(file, "r") as io:
            read_nwbfile = io.read()
            traces = np.array(
                read_nwbfile.processing["CalciumActivity"]
                .data_interfaces["SignalRawFluor"]
                .roi_response_series["SignalCalciumImResponseSeries"]
                .data
            )
            neuron_ids = np.array(
                read_nwbfile.processing["CalciumActivity"].data_interfaces["NeuronIDs"].labels,
                dtype=np.dtype(str),
            )
            # sampling frequency is 4 Hz
            time_vector = np.arange(0, traces.shape[0]).astype(np.dtype(float)) / 4

        return neuron_ids, traces, time_vector

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(citation="Nejatbakhsh 2020 dataset")
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Nejatbakhsh et al., 2020 neural data and saves it as a pickle file

        The data is read from NWB files located in the dataset directory.

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the subfolders and files in the dataset directory:
                - Extract neuron IDs, calcium traces, and time vector from each NWB file.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        preprocessed_data = dict()
        worm_idx = 0
        for tree in tqdm(os.listdir(os.path.join(self.raw_data_path, self.source_dataset))):
            print(f"DEBUG tree {tree}")
            if tree.startswith('.'):
                continue
            subfolder = os.path.join(self.raw_data_path, self.source_dataset, tree)
            if not os.path.isdir(subfolder):
                continue
            for file_name in os.listdir(subfolder):
                if not file_name.endswith('.nwb'):
                    continue
                neuron_ids, traces, raw_time_vector = self.extract_data(
                    os.path.join(self.raw_data_path, self.source_dataset, subfolder, file_name)
                )
                preprocessed_data, worm_idx = self.preprocess_traces(
                    [neuron_ids], [traces], [raw_time_vector], preprocessed_data, worm_idx
                )
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Venkatachalam2024Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Venkatachalam et al., 2024 connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Venkatachalam et al., 2024 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        unzip_and_extract_csv(source_directory, zip_path): Unzips the provided ZIP file and extracts the CSV file.
        load_data(file_name): Loads the data from the extracted CSV file.
        extract_data(data): Extracts neuron IDs, calcium traces, and time vector from the CSV data.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Venkatachalam et al., 2024 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Venkatachalam2024Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Venkatachalam2024",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def unzip_and_extract_csv(self, source_directory, zip_path):
        """
        Unzips the provided ZIP file and extracts the CSV file.

        Parameters:
            source_directory (str): The directory where the ZIP file is located.
            zip_path (str): The path to the ZIP file.

        Returns:
            str: The path to the extracted CSV file.
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            path = zip_ref.extractall(source_directory)
        return zip_path.replace(".zip", ".csv")

    def load_data(self, file_name):
        """
        Loads the data from the extracted CSV file.

        Parameters:
            file_name (str): The name of the ZIP file containing the CSV data.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        zip_path = os.path.join(self.raw_data_path, self.source_dataset, file_name)
        csv_file = self.unzip_and_extract_csv(
            os.path.join(self.raw_data_path, self.source_dataset), zip_path
        )
        data = pd.read_csv(csv_file)
        return data

    def extract_data(self, data):
        """
        Extracts neuron IDs, calcium traces, and time vector from the CSV data.

        Parameters:
            data (pd.DataFrame): The loaded data as a pandas DataFrame.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        neuron_ids = data["neuron"].unique()
        # 9 + 98 blank timesteps at beginning (0-97)
        time_vector = (
            data.columns[107:-1].astype(float).to_numpy()
        )  # Assuming columns 9 onwards are time points
        traces = data.iloc[:, 107:-1].values.T  # Transpose to get (time, neurons)

        return neuron_ids, traces, time_vector

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(citation="Venkatachalam dataset")
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Venkatachalam et al., 2024 neural data and saves it as a pickle file

        The data is read from ZIP files containing CSV data located in the dataset directory.

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the ZIP files in the dataset directory:
                - Unzip and extract the CSV file.
                - Load the data from the CSV file.
                - Extract neuron IDs, calcium traces, and time vector from the CSV data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        preprocessed_data = dict()
        worm_idx = 0
        for file_name in os.listdir(os.path.join(self.raw_data_path, self.source_dataset)):
            if not file_name.endswith(".zip"):
                continue
            raw_data = self.load_data(file_name)
            neuron_ids, traces, raw_time_vector = self.extract_data(raw_data)
            preprocessed_data, worm_idx = self.preprocess_traces(
                [neuron_ids], [traces], [raw_time_vector], preprocessed_data, worm_idx
            )
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Kato2015Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Kato et al., 2015 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Kato et al., 2015 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Kato et al., 2015 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Kato2015Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Kato2015",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
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
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Kato et al., Cell 2015, _Global Brain Dynamics Embed the Motor Command Sequence of Caenorhabditis elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Kato et al., 2015 neural data and saves it as a pickle file.

        The data is read from MAT files named "WT_Stim.mat" and "WT_NoStim.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
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


class Nichols2017Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Nichols et al., 2017 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Nichols et al., 2017 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Nichols et al., 2017 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Nichols2017Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Nichols2017",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]
        # Time vector in seconds
        timeVectorSeconds = arr["timeVectorSeconds"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Nichols et al., Science 2017, _A global brain state underlies C. elegans sleep behavior_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Nichols et al., 2017 neural data and saves it as a pickle file.

        The data is read from MAT files named "n2_let.mat", "n2_prelet.mat", "npr1_let.mat", and "npr1_prelet.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
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


class Skora2018Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Skora et al., 2018 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Skora et al., 2018 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Skora et al., 2018 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Skora2018Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Skora2018",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]
        # Time vector in seconds
        timeVectorSeconds = arr["timeVectorSeconds"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Skora et al., Cell Reports 2018, _Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Skora et al., 2018 neural data and saves it as a pickle file.

        The data is read from MAT files named "WT_fasted.mat" and "WT_starved.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
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


class Kaplan2020Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Kaplan et al., 2020 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Kaplan et al., 2020 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Kaplan et al., 2020 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Kaplan2020Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Kaplan2020",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Load data with mat73
        data = mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["neuron_ID"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces_bleach_corrected"]
        # Time vector in seconds
        timeVectorSeconds = arr["time_vector"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Kaplan et al., Neuron 2020, _Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Kaplan et al., 2020 neural data and saves it as a pickle file.

        The data is read from MAT files named "Neuron2019_Data_MNhisCl_RIShisCl.mat", "Neuron2019_Data_RIShisCl.mat", and "Neuron2019_Data_SMDhisCl_RIShisCl.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
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


class Yemini2021Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Yemini et al., 2021 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Yemini et al., 2021 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(raw_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Yemini et al., 2021 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Yemini2021Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Yemini2021",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Overriding the base class method to use scipy.io.loadmat for .mat files
        data = loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, raw_data):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            raw_data (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Extract frames per second (fps) from the raw data.
            2. Extract the list of files, bilateral neurons, and boolean masks for left/right neurons.
            3. Extract histogram-normalized neuronal traces.
            4. Initialize lists for neuron IDs, traces, and time vectors.
            5. Iterate through each file in the list of files:
                - Initialize lists for neurons, activity, and time vector for the current file.
                - Iterate through each neuron in the list of bilateral neurons:
                    - Assign neuron names with L/R and get associated traces.
                    - Handle non-bilaterally symmetric neurons.
                    - Handle bilaterally symmetric neurons and assign left/right traces.
                    - Update the time vector if necessary.
                - Append the neurons, activity, and time vector for the current file to the respective lists.
            6. Return the extracted neuron IDs, traces, and time vectors.
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
            if np.isnan(real_data).any():
                real_data = imputer.fit_transform(real_data)
            # Observed empirically that the first three values of activity equal 0.0s
            activity = activity[4:]
            tvec = tvec[4:]
            # Add acitvity to list of traces
            traces.append(activity)
            # Add time vector to list of time vectors
            time_vector_seconds.append(tvec)
        # Return the extracted data
        return neuron_IDs, traces, time_vector_seconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Yemini et al., Cell CurrBio 2022, _NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Yemini et al., 2021 neural data and saves it as a pickle file.

        The data is read from MAT files named "Head_Activity_OH15500.mat", "Head_Activity_OH16230.mat", and "Tail_Activity_OH16230.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
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


class Uzel2022Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Uzel et al., 2022 connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Uzel et al., 2022 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(save_as="graph_tensors_uzel2022.pt"): Preprocesses the Uzel et al., 2022 neural data and saves is as a file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Uzel2022Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Uzel2022",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Load data with mat73
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]  # (time, neurons)
        # Time vector in seconds
        timeVectorSeconds = arr["tv"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Uzel et al., Cell CurrBio 2022, _A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Uzel et al., 2022 neural data and saves it as a pickle file.

        The data is read from a MAT file named "Uzel_WT.mat".

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_uzel2022.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Load the data from the MAT file.
            3. Extract neuron IDs, calcium traces, and time vector from the loaded data.
            4. Preprocess the traces and update the preprocessed data dictionary.
            5. Reshape the calcium data for each worm.
            6. Save the preprocessed data to the specified file.
        """
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


class Dag2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Dag et al., 2023 connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Dag et al., 2023 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        load_data(file_name): Loads the data from the specified HDF5 file.
        load_labels_dict(labels_file="NeuroPAL_labels_dict.json"): Loads the neuron labels dictionary from a JSON file.
        find_nearest_label(query, possible_labels, char="?"): Finds the nearest neuron label from a list given a query.
        extract_data(data_file, labels_file): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(save_as="graph_tensors_dag2023.pt"): Preprocesses the Dag et al., 2023 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Dag2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Dag2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        """
        Loads the data from the specified HDF5 file.

        Parameters:
            file_name (str): The name of the HDF5 file containing the data.

        Returns:
            h5py.File: The loaded data as an HDF5 file object.
        """
        data = h5py.File(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r")
        return data

    def load_labels_dict(self, labels_file="NeuroPAL_labels_dict.json"):
        """
        Loads the neuron labels dictionary from a JSON file.

        Parameters:
            labels_file (str, optional): The name of the JSON file containing the neuron labels. Default is "NeuroPAL_labels_dict.json".

        Returns:
            dict: The loaded neuron labels dictionary.
        """
        with open(os.path.join(self.raw_data_path, self.source_dataset, labels_file), "r") as f:
            label_info = json.load(f)
        return label_info

    def find_nearest_label(self, query, possible_labels, char="?"):
        """
        Finds the nearest neuron label from a list given a query.

        Parameters:
            query (str): The query string containing the neuron label with ambiguity.
            possible_labels (list): The list of possible neuron labels.
            char (str, optional): The character representing ambiguity in the query. Default is "?".

        Returns:
            tuple: A tuple containing the nearest neuron label and its index in the possible labels list.
        """
        # Ensure the possible labels is a sorted list
        possible_labels = sorted(possible_labels)
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

    def extract_data(self, data_file, labels_file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data file.

        Parameters:
            data_file (str): The path to the HDF5 data file.
            labels_file (str): The path to the JSON file containing neuron labels.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Load the data file and labels file.
            2. Extract the mapping of indices in the data to neuron labels.
            3. Extract neural activity traces and time vector.
            4. Get neuron labels corresponding to indices in calcium data.
            5. Handle ambiguous neuron labels.
            6. Return the extracted data.
        """
        # Load the data file and labels file
        file_data = self.load_data(data_file)
        label_info = self.load_labels_dict(labels_file)
        # Extract the mapping of indices in the data to neuron labels
        index_map, _ = label_info.get(data_file.split("/")[-1].strip("-data.h5"), (dict(), None))
        # Neural activity traces
        calcium = np.array(file_data["gcamp"]["traces_array_F_F20"])  # (time, neurons)
        # Time vector in seconds
        timevec = np.array(file_data["timing"]["timestamp_confocal"])[: calcium.shape[0]]  # (time,)
        # Get neuron labels corresponding to indices in calcium data
        indices = []
        neurons = []
        # If there is an index map, use it to extract the labeled neurons
        if index_map:
            # Indices in index_map correspond to named neurons
            for calnum in index_map:
                # NOTE: calnum is a string, not an integer
                assert (
                    int(calnum) <= calcium.shape[1]
                ), f"Index out of range. calnum: {calnum}, calcium.shape[1]: {calcium.shape[1]}"
                lbl = index_map[calnum]["label"]
                neurons.append(lbl)
                # Need to minus one because Julia index starts at 1 whereas Python index starts with 0
                idx = int(calnum) - 1
                indices.append(idx)
            # Remaining indices correspond to unknown neurons
            for i in range(calcium.shape[1]):
                if i not in set(indices):
                    indices.append(i)
                    neurons.append(str(i))
        # Otherwise, use the indices as the neuron labels for all traces
        else:
            indices = list(range(calcium.shape[1]))
            neurons = [str(i) for i in indices]
        # Ensure only calcium data at selected indices is kept
        calcium = calcium[:, indices]
        # Neurons with dorso-ventral/lateral ambiguity have a '?' in the label that must be inferred
        neurons_copy = []
        for label in neurons:
            # If the neuron is unknown it will have a numeric label corresponding to its index
            if label.isnumeric():
                neurons_copy.append(label)
                continue
            # Look for the closest neuron label that will match the current string containing '?'
            replacement, _ = self.find_nearest_label(
                label, set(NEURON_LABELS) - set(neurons_copy), char="?"
            )
            neurons_copy.append(replacement)
        # Convert the list of neuron labels to a numpy array
        neurons = np.array(neurons_copy, dtype=str)
        # Make the extracted data into a list of arrays
        all_IDs = [neurons]
        all_traces = [calcium]
        timeVectorSeconds = [timevec]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Dag et al., 2023 neural data and saves it as a pickle file.

        The data is read from HDF5 files located in the dataset directory.

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the subfolders and files in the dataset directory:
                - Extract neuron IDs, calcium traces, and time vector from each HDF5 file.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Load and preprocess data
        preprocessed_data = dict()  # Store preprocessed data
        worm_idx = 0  # Initialize worm index outside file loop
        # There are two subfolders in the Dag2023 dataset: 'swf415_no_id' and 'swf702_with_id'
        withid_data_files = os.path.join(self.raw_data_path, "Dag2023", "swf702_with_id")
        noid_data_files = os.path.join(self.raw_data_path, "Dag2023", "swf415_no_id")
        # 'NeuroPAL_labels_dict.json' maps data file names to a dictionary of neuron label information
        labels_file = "NeuroPAL_labels_dict.json"
        # First deal with the swf702_with_id which contains data from labeled neurons
        for file in os.listdir(withid_data_files):
            if not file.endswith(".h5"):
                continue
            data_file = os.path.join("swf702_with_id", file)
            neurons, raw_traces, time_vector_seconds = self.extract_data(data_file, labels_file)
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons, raw_traces, time_vector_seconds, preprocessed_data, worm_idx
            )  # preprocess
        ### DEBUG ###
        # Next deal with the swf415_no_id which contains purely unlabeled neuron data
        # NOTE: These don't get used at all as they are skipped in
        # NeuralBasePreprocessor.preprocess_traces, because num_named_neurons == 0.
        for file in os.listdir(noid_data_files):
            if not file.endswith(".h5"):
                continue
            data_file = os.path.join("swf415_no_id", file)
            neurons, raw_traces, time_vector_seconds = self.extract_data(data_file, labels_file)
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons, raw_traces, time_vector_seconds, preprocessed_data, worm_idx
            )  # preprocess
        ### DEBUG ###
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")
        return None

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Dag et al., Cell 2023. “_Dissecting the Functional Organization of the C. Elegans Serotonergic System at Whole-Brain Scale_"
        )
        return extra_info


class Flavell2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Flavell et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Flavell et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        find_nearest_label(query, possible_labels, char="?"): Finds the nearest neuron label from a list given a query.
        load_data(file_name): Loads the data from the specified HDF5 or JSON file.
        extract_data(file_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(save_as="graph_tensors_flavell2023.pt"): Preprocesses the Flavell et al., 2023 neural data and saves is as pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Flavell2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Flavell2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def find_nearest_label(self, query, possible_labels, char="?"):
        """
        Finds the nearest neuron label from a list given a query.

        Parameters:
            query (str): The query string containing the neuron label with ambiguity.
            possible_labels (list): The list of possible neuron labels.
            char (str, optional): The character representing ambiguity in the query. Default is "?".

        Returns:
            tuple: A tuple containing the nearest neuron label and its index in the possible labels list.
        """
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
        """
        Loads the data from the specified HDF5 or JSON file.

        Parameters:
            file_name (str): The name of the HDF5 or JSON file containing the data.

        Returns:
            dict or h5py.File: The loaded data as a dictionary or HDF5 file object.
        """
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
        Extracts neuron IDs, calcium traces, and time vector from the loaded data file.

        Parameters:
            file_data (dict or h5py.File): The loaded data file.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Ensure the file data is in the expected format (dict or h5py.File).
            2. Extract the time vector in seconds.
            3. Extract raw traces and initialize the calcium data array.
            4. Extract neuron labels and handle ambiguous neuron labels.
            5. Filter for unique neuron labels and get data for unique neurons.
            6. Return the extracted data.
        """
        # The files are expected to use a JSON or H5 format
        assert isinstance(file_data, (dict, h5py.File)), f"Unsupported data type: {type(file_data)}"
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
        # Return the extracted data
        return neurons, calcium_data, time_in_seconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Atanas et al., Cell 2023, _Brain-Wide Representations of Behavior Spanning Multiple Timescales and States in C. Elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Flavell et al., 2023 neural data and saves it as a pickle file.

        The data is read from HDF5 or JSON files located in the dataset directory.

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_flavell2023.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data.
            2. Iterate through the files in the dataset directory:
                - Load the data from the file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # TODO: Encapsulate the single worm part of this method into a `preprocess_traces` method.
        # Load and preprocess data
        preprocessed_data = dict()
        for i, file in enumerate(os.listdir(os.path.join(self.raw_data_path, self.source_dataset))):
            if not (file.endswith(".h5") or file.endswith(".json")):
                continue
            worm = "worm" + str(i)
            file_data = self.load_data(file)  # load
            # Extract raw data
            neurons, calcium_data, time_in_seconds = self.extract_data(file_data)
            # Set time to start at 0.0 seconds
            time_in_seconds = time_in_seconds - time_in_seconds[0]  # vector
            # Map named neurons
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(neurons)
            # Normalize calcium data
            calcium_data = self.normalize_data(calcium_data)  # matrix
            # Compute calcium dynamics (residual calcium)
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


class Leifer2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Leifer et al., 2023 connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Leifer et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        str_to_float(str_num): Converts a string in scientific notation to a floating-point number.
        load_labels(file_path): Loads neuron labels from a text file.
        load_time_vector(file_path): Loads the time vector from a text file.
        load_data(file_path): Loads the neural data from a text file.
        is_monotonic_linear(arr): Checks if the array is a line with a constant slope (i.e., linear).
        filter_bad_traces_by_linear_segments(data, window_size=50, linear_segment_threshold=1e-3): Filters out traces with significant proportions of linear segments.
        create_neuron_idx(label_list): Creates a mapping of neuron labels to indices.
        extract_data(data_file, labels_file, time_file): Extracts neuron IDs, calcium traces, and time vector from the loaded data files.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(save_as="graph_tensors_leifer2023.pt"): Preprocesses the Leifer et al., 2023 neural data and saves it as a pickle a file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Leifer2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
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
        Converts a string in scientific notation to a floating-point number.

        Parameters:
            str_num (str): The string in scientific notation.

        Returns:
            float: The converted floating-point number.
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
        """
        Loads neuron labels from a text file.

        Parameters:
            file_path (str): The path to the text file containing neuron labels.

        Returns:
            list: A list of neuron labels.
        """
        with open(file_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines()]
        return labels

    def load_time_vector(self, file_path):
        """
        Loads the time vector from a text file.

        Parameters:
            file_path (str): The path to the text file containing the time vector.

        Returns:
            np.ndarray: The time vector as a numpy array.
        """
        with open(file_path, "r") as f:
            timeVectorSeconds = [self.str_to_float(line.strip("\n")) for line in f.readlines()]
            timeVectorSeconds = np.array(timeVectorSeconds, dtype=np.float32).reshape(-1, 1)
        return timeVectorSeconds

    def load_data(self, file_path):
        """
        Loads the neural data from a text file.

        Parameters:
            file_path (str): The path to the text file containing the neural data.

        Returns:
            np.ndarray: The neural data as a numpy array.
        """
        with open(file_path, "r") as f:
            data = [list(map(float, line.split(" "))) for line in f.readlines()]
        data_array = np.array(data, dtype=np.float32)
        return data_array

    def is_monotonic_linear(self, arr):
        """
        Checks if the array is a line with a constant slope (i.e., linear).

        Parameters:
            arr (np.ndarray): The input array to check.

        Returns:
            bool: True if the array is linear, False otherwise.
        """
        assert arr.ndim == 1, "Array must be a 1D (univariate) time series."
        diff = np.round(np.diff(arr), decimals=3)
        result = np.unique(diff)
        return result.size == 1

    def filter_bad_traces_by_linear_segments(
        self, data, window_size=50, linear_segment_threshold=1e-3
    ):
        """
        Filters out traces with significant proportions of linear segments. Linear segments suggest
        that the data was imputed with linear interpolation to remove stretches of NaN values.

        There are weird-looking traces in the Leifer2023 raw data caused by interpolations of missing values
        (NaNs) when neurons were not consistently tracked over time due to imperfect nonrigid registration.
        This helper function was written to filter out these problematic imputed neural traces.

        Parameters:
            data (np.array): The neural data array with shape (time_points, neurons).
            window_size (int): The size of the window to check for linearity.
            linear_segment_threshold (float): Proportion of linear segments above which traces are considered bad.

        Returns:
            tuple: Tuple of filtered neural data and the associated mask into the original data array.
        """
        t, n = data.shape
        linear_segments = np.zeros(n, dtype=int)
        window_start = range(
            0, t - window_size, window_size // 2
        )  # non-overlapping or staggered windows (faster)
        for i in window_start:
            segment = data[i : i + window_size, :]
            ls = np.apply_along_axis(self.is_monotonic_linear, 0, segment)
            linear_segments += ls.astype(int)
        proportion_linear = linear_segments / len(window_start)
        bad_traces_mask = np.array(proportion_linear > linear_segment_threshold)
        good_traces_mask = ~bad_traces_mask
        filtered_data = data[:, good_traces_mask]
        return filtered_data, good_traces_mask

    def create_neuron_idx(self, label_list):
        """
        Creates a mapping of neuron labels to indices.

        Parameters:
            label_list (list): The list of neuron labels.

        Returns:
            tuple: A tuple containing the neuron-to-index mapping and the number of named neurons.
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
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data files.

        Parameters:
            data_file (str): The path to the text file containing the neural data.
            labels_file (str): The path to the text file containing neuron labels.
            time_file (str): The path to the text file containing the time vector.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Load the neuron labels, time vector, and neural data.
            2. Filter out bad traces based on linear segments.
            3. Return the extracted data.
        """
        real_data = self.load_data(data_file)  # shaped (time, neurons)
        # In some strange cases there are more labels than neurons
        label_list = self.load_labels(labels_file)[: real_data.shape[1]]
        time_in_seconds = self.load_time_vector(time_file)
        # Check that the data, labels and time shapes match
        assert real_data.shape[1] == len(
            label_list
        ), f"Data and labels do not match!\n Files: {data_file}, {labels_file}"
        assert (
            real_data.shape[0] == time_in_seconds.shape[0]
        ), f"Time vector does not match data!\n Files: {data_file}, {time_file}"
        # Remove neuron traces that are all NaN values
        mask = np.argwhere(~np.isnan(real_data).all(axis=0)).flatten()
        real_data = real_data[:, mask]
        label_list = np.array(label_list, dtype=str)[mask].tolist()
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(real_data).any():
            real_data = imputer.fit_transform(real_data)
        # Remove badly imputed neurons from the data
        filt_real_data, filt_mask = self.filter_bad_traces_by_linear_segments(real_data)
        filt_label_list = np.array(label_list, dtype=str)[filt_mask].tolist()
        # Return the extracted data
        return filt_label_list, filt_real_data, time_in_seconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Randi et al., Nature 2023, _Neural Signal Propagation Atlas of Caenorhabditis Elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Leifer et al., 2023 neural data and saves it as a pickle file.

        The data is read from text files located in the dataset directory.

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_leifer2023.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data.
            2. Iterate through the files in the dataset directory:
                - Load the data from the files.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to the specified file.
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
            # Map named neurons
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(label_list)
            if num_named_neurons == 0:  # skip worms with no labelled neuron
                worm_idx -= 1
                continue
            # Normalize calcium data
            calcium_data = self.normalize_data(real_data)  # matrix
            # Compute calcium dynamics (residual calcium)
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


class Lin2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Lin et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Lin et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(file_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Lin et al., 2023 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Lin2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Lin2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Overriding the base class method to use scipy.io.loadmat for .mat files
        data = loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, data_file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data file.

        Parameters:
            data_file (dict): The loaded data file.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Ensure the file data is in the expected format (dict).
            2. Extract the time vector in seconds.
            3. Extract raw traces and initialize the calcium data array.
            4. Extract neuron labels and handle ambiguous neuron labels.
            5. Filter for unique neuron labels and get data for unique neurons.
            6. Return the extracted data.
        """
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
        # NOTE: This is very slow with the default settings on this dataset!
        imputer = IterativeImputer(random_state=0)
        if np.isnan(real_data).any():
            real_data = imputer.fit_transform(real_data)
        # Make the extracted data into a list of arrays
        neuron_IDs, raw_traces, time_vector_seconds = [neurons], [raw_activitiy], [raw_time_vec]
        # Return the extracted data
        return neuron_IDs, raw_traces, time_vector_seconds

    def create_metadata(self):
        """
        Creates a dictionary of extra information or metadata for the dataset.

        Returns:
            dict: A dictionary containing extra information or metadata.
        """
        extra_info = dict(
            citation="Lin et al., Science Advances 2023, _Functional Imaging and Quantification of Multineuronal Olfactory Responses in C. Elegans_"
        )
        return extra_info

    def preprocess(self):
        """
        Preprocesses the Lin et al., 2023 neural data and saves it as a pickle file.

        The data is read from MAT files located in the dataset directory.

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_lin2023.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data.
            2. Iterate through the files in the dataset directory:
                - Load the data from the file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to the specified file.
        """
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


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
