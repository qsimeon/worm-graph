from preprocess._pkg import *

# Initialize logger
logger = logging.getLogger(__name__)


### Function definitions # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def pickle_neural_data(
#     url,
#     zipfile,
#     source_dataset="all",
#     # TODO: Try different transforms from sklearn such as QuantileTransformer, etc. as well as custom CausalNormalizer.
#     transform=StandardScaler(),  # StandardScaler() #PowerTransformer() #CausalNormalizer() #None
#     smooth_method="none",
#     interpolate_method="linear",
#     resample_dt=None,
#     cleanup=False,
#     **kwargs,
# ):
#     """Preprocess and save C. elegans neural data to .pickle format.

#     This function downloads and extracts the open-source datasets if not found in the
#     root directory, preprocesses the neural data using the corresponding DatasetPreprocessor class,
#     and then saves it to .pickle format. The processed data is saved in the
#     data/processed/neural folder for further use.

#     Args:
#         url (str): Download link to a zip file containing the open-source data in raw form.
#         zipfile (str): The name of the zipfile that is being downloaded.
#         source_dataset (str, optional): The name of the source dataset to be pickled.
#             If None or 'all', all datasets are pickled. Default is 'all'.
#         transform (object, optional): The sklearn transformation to be applied to the data.
#             Default is StandardScaler().
#         smooth_method (str, optional): The smoothing method to apply to the data;
#             options are 'gaussian', 'exponential', or 'moving'. Default is 'moving'.
#         interpolate_method (str, optional): The scipy interpolation method to use when resampling the data.
#             Default is 'linear'.
#         resample_dt (float, optional): The resampling time interval in seconds.
#             If None, no resampling is performed. Default is None.
#         cleanup (bool, optional): If True, deletes the unzipped folder after processing. Default is False.
#         **kwargs: Additional keyword arguments to be passed to the DatasetPreprocessor class.

#     Returns:
#         None

#     Raises:
#         AssertionError: If an invalid source dataset is requested.
#         NameError: If the specified preprocessor class is not found.


#     Steps:
#         1. Construct paths for the zip file and source data.
#         2. Create the neural data directory if it doesn't exist.
#         3. Download and extract the zip file if the source data is not found.
#         4. Instantiate and use the appropriate DatasetPreprocessor class to preprocess the data.
#         5. Save the preprocessed data to .pickle format.
#         6. Optionally, delete the unzipped folder if cleanup is True.
#     """
#     zip_path = os.path.join(ROOT_DIR, zipfile)
#     source_path = os.path.join(ROOT_DIR, zipfile.strip(".zip"))
#     # Make the neural data directory if it doesn't exist
#     processed_path = os.path.join(ROOT_DIR, "data/processed/neural")
#     if not os.path.exists(processed_path):
#         os.makedirs(processed_path, exist_ok=True)
#     # If .zip not found in the root directory, download the curated open-source worm datasets
#     if not os.path.exists(source_path):
#         download_url(url=url, folder=ROOT_DIR, filename=zipfile)
#         # Extract all the datasets ... OR
#         if source_dataset.lower() == "all":
#             # Extract zip file then delete it
#             extract_zip(zip_path, folder=source_path, delete_zip=True)
#         # Extract just the requested source dataset
#         else:
#             bash_command = [
#                 "unzip",
#                 zip_path,
#                 "{}/*".format(source_dataset),
#                 "-d",
#                 source_path,
#                 "-x",
#                 "__MACOSX/*",
#             ]
#             # Run the bash command
#             std_out = subprocess.run(bash_command, text=True)
#             # Output to log or terminal
#             logger.info(f"Unzip status {std_out} ...")
#             # Delete the zip file
#             os.unlink(zip_path)
#     # (re)-Pickle all the datasets ... OR
#     if source_dataset is None or source_dataset.lower() == "all":
#         for source in EXPERIMENT_DATASETS:
#             logger.info(f"Start processing {source}.")
#             try:
#                 # Instantiate the relevant preprocessor class
#                 preprocessor = eval(source + "Preprocessor")(
#                     transform,
#                     smooth_method,
#                     interpolate_method,
#                     resample_dt,
#                     **kwargs,
#                 )
#                 # Call its method
#                 preprocessor.preprocess()
#             except NameError as e:
#                 logger.info(f"NameError calling proprocessor: {e}")
#                 continue
#         # Create a file to indicate that the preprocessing was successful
#         open(os.path.join(processed_path, ".processed"), "a").close()
#     # ... (re)-Pickle a single dataset
#     else:
#         assert (
#             source_dataset in EXPERIMENT_DATASETS
#         ), "Invalid source dataset requested! Please pick one from:\n{}".format(
#             list(EXPERIMENT_DATASETS)
#         )
#         logger.info(f"Start processing {source_dataset}.")
#         try:
#             # Instantiate the relevant preprocessor class
#             preprocessor = eval(source_dataset + "Preprocessor")(
#                 transform,
#                 smooth_method,
#                 interpolate_method,
#                 resample_dt,
#                 **kwargs,
#             )
#             # Call its method
#             preprocessor.preprocess()
#         except NameError:
#             pass
#     # Delete the unzipped folder
#     if cleanup:
#         shutil.rmtree(source_path)
#     return None
def process_single_dataset(args):
    """Helper function to process a single dataset

    Args:
        args (tuple): (source, transform, smooth_method, interpolate_method, resample_dt, kwargs)
    """
    source, transform, smooth_method, interpolate_method, resample_dt, kwargs = args
    try:
        logger.info(f"Start processing {source}.")
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
        return True
    except NameError as e:
        logger.info(f"NameError calling preprocessor: {e}")
        return False


def pickle_neural_data(
    url,
    zipfile,
    source_dataset="all",
    transform=StandardScaler(),
    smooth_method="none",
    interpolate_method="linear",
    resample_dt=None,
    cleanup=False,
    n_workers=None,  # New parameter for controlling number of workers
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
        try:
            download_url(url=url, folder=ROOT_DIR, filename=zipfile)
        except Exception as e:
            logger.error(f"Failed to download using async method: {e}")
            logger.info("Falling back to wget...")
            # Fallback to wget if async download fails
            import subprocess

            subprocess.run(
                [
                    "wget",
                    "-O",
                    os.path.join(ROOT_DIR, zipfile),
                    "--tries=3",  # Retry 3 times
                    "--continue",  # Resume partial downloads
                    "--progress=bar:force",  # Show progress bar
                    url,
                ]
            )
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
            logger.info(f"Unzip status {std_out} ...")
            # Delete the zip file
            os.unlink(zip_path)
    # (re)-Pickle all the datasets ... OR
    if source_dataset is None or source_dataset.lower() == "all":
        # Determine number of workers (use CPU count - 1 by default)
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        # Prepare arguments for parallel processing
        process_args = [
            (source, transform, smooth_method, interpolate_method, resample_dt, kwargs)
            for source in EXPERIMENT_DATASETS
        ]

        # Use multiprocessing Pool to process datasets in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_dataset, process_args)

        # Create a file to indicate that the preprocessing was successful
        if any(results):  # If at least one dataset was processed successfully
            open(os.path.join(processed_path, ".processed"), "a").close()

    # ... (re)-Pickle a single dataset
    else:
        assert (
            source_dataset in EXPERIMENT_DATASETS
        ), "Invalid source dataset requested! Please pick one from:\n{}".format(
            list(EXPERIMENT_DATASETS)
        )
        process_single_dataset(
            (source_dataset, transform, smooth_method, interpolate_method, resample_dt, kwargs)
        )

    # Delete the unzipped folder
    if cleanup:
        shutil.rmtree(source_path)
    return None


def get_presaved_datasets(url, file):
    """Download and unzip presaved data patterns.

    This function downloads and extracts presaved data patterns).
    from the specified URL. The extracted data is saved in the 'data' folder.
    The zip file is deleted after extraction.

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


def preprocess_connectome(raw_files, source_connectome=None):
    """Convert the raw connectome data to a graph tensor.

    This function processes raw connectome data, which includes chemical
    synapses and gap junctions, into a format suitable for use in machine
    learning or graph analysis. It reads the raw data in tabular format (.csv, .xls[x]),
    processes it to extract the relevant information, and creates graph
    tensors that represent the C. elegans connectome. The resulting
    graph tensors are saved in the 'data/processed/connectome' folder
    as 'graph_tensors.pt'. We distinguish between electrical (gap junction)
    and chemical synapses by using an edge attribute tensor with two feature dimensions:
    the first feature represents the weight of the gap junctions; and the second feature
    represents the weight of the chemical synapses.

    Args:
        raw_files (list): Contain the names of the raw connectome data to preprocess.
        source_connectome (str, optional): The source connectome file to use for preprocessing. Options include:
            - "openworm": OpenWorm project  (augmentation of earlier connectome with neurotransmitter type)
            - "funconn" or "randi_2023": Randi et al., 2023 (functional connectivity)
            - "witvliet_7": Witvliet et al., 2020 (adult 7)
            - "witvliet_8": Witvliet et al., 2020 (adult 8)
            - "white_1986_whole": White et al., 1986 (whole)
            - "white_1986_n2u": White et al., 1986 (N2U)
            - "white_1986_jsh": White et al., 1986 (JSH)
            - "white_1986_jse": White et al., 1986 (JSE)
            - "cook_2019": Cook et al., 2019
            - "all": preprocess all of the above connectomes separately
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
    * A connectome is a comprehensive map of the neural connections within an
      organism's brain or nervous system. It is essentially the wiring diagram
      of the brain, detailing how neurons and their synapses are interconnected.
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

    # Make the connectome data directory if it doesn't exist
    processed_path = os.path.join(ROOT_DIR, "data/processed/connectome")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path, exist_ok=True)

    # Determine appropriate preprocessing class based on publication
    preprocessors = {
        "openworm": OpenWormPreprocessor,
        "chklovskii": ChklovskiiPreprocessor,
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

    # Preprocess all the connectomes including the default one
    if source_connectome == "all":
        for preprocessor_class in preprocessors.values():
            preprocessor_class().preprocess()
        # Create a file to indicate that the preprocessing was successful
        open(os.path.join(processed_path, ".processed"), "a").close()

    # Preprocess just the requested connectome
    else:
        preprocessor_class = preprocessors.get(source_connectome, DefaultPreprocessor)
        preprocessor_class().preprocess()

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
        save_graph_tensors(save_as: str, graph, num_classes, node_type, node_label, node_index, node_class):
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
        that represent the connectome. It ensures the correct mapping of neurons to their classes
        and types, checks for the symmetry of the gap junction adjacency matrix, and adds
        missing nodes with zero-weight edges to maintain graph completeness.

        Args:
            edge_index (torch.Tensor): Tensor containing the edge indices.
            edge_attr (torch.Tensor): Tensor containing the edge attributes.

        Returns:
            graph (torch_geometric.data.Data): The processed graph data object.
            node_type (torch.Tensor): Tensor of integers representing neuron types.
            node_label (dict): Dictionary mapping node indices to neuron labels.
            node_index (torch.Tensor): Tensor containing the node indices.
            node_class (dict): Dictionary mapping node indices to neuron classes.
            num_classes (int): The number of unique neuron classes.
        """
        # Filter the neuron master sheet to include only neurons present in the labels
        df_master = self.neuron_master_sheet[
            self.neuron_master_sheet["label"].isin(self.neuron_labels)
        ]

        # Create a position dictionary (pos) for neurons using their x, y, z coordinates
        pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
        pos = {
            self.neuron_to_idx[label]: [
                pos_dict[label]["x"],
                pos_dict[label]["y"],
                pos_dict[label]["z"],
            ]
            for label in pos_dict
        }

        # Encode the neuron class (e.g., ADA, ADF) and create a mapping from node index to neuron class
        df_master["class"] = df_master["class"].fillna("Unknown")
        node_class = {
            self.neuron_to_idx[label]: neuron_class
            for label, neuron_class in zip(df_master["label"], df_master["class"])
        }
        num_classes = len(df_master["class"].unique())

        # Alphabetically sort neuron types and encode them as integers
        df_master["type"] = df_master["type"].fillna("Unknown")
        unique_types = sorted(df_master["type"].unique()) # inter, motor, pharynx, sensory
        type_to_int = {neuron_type: i for i, neuron_type in enumerate(unique_types)}

        # Create tensor of neuron types (y) using the encoded integers
        y = torch.tensor(
            [type_to_int[neuron_type] for neuron_type in df_master["type"]], dtype=torch.long
        )

        # Map node indices to neuron types using integers
        node_type = {
            self.neuron_to_idx[label]: type_to_int[neuron_type]
            for label, neuron_type in zip(df_master["label"], df_master["type"])
        }

        # Initialize the node features (x) as a tensor, here set as empty with 1024 features per node (customize as needed)
        x = torch.empty(len(self.neuron_labels), 1024, dtype=torch.float)

        # Create the mapping from node indices to neuron labels (e.g., 'ADAL', 'ADAR', etc.)
        node_label = {idx: label for label, idx in self.neuron_to_idx.items()}

        # Create the node index tensor for the graph
        node_index = torch.arange(len(self.neuron_labels))

        # Add missing nodes with zero-weight edges to ensure the adjacency matrix is 300x300
        all_indices = torch.arange(len(self.neuron_labels))
        full_edge_index = torch.combinations(all_indices, r=2).T
        existing_edges_set = set(map(tuple, edge_index.T.tolist()))

        additional_edges = []
        additional_edge_attr = []
        for edge in full_edge_index.T.tolist():
            if tuple(edge) not in existing_edges_set:
                additional_edges.append(edge)
                additional_edge_attr.append(
                    [0, 0]
                )  # Add a zero-weight edge for missing connections

        # If there are additional edges, add them to the edge_index and edge_attr tensors
        if additional_edges:
            additional_edges = torch.tensor(additional_edges).T
            additional_edge_attr = torch.tensor(additional_edge_attr, dtype=torch.float)
            edge_index = torch.cat([edge_index, additional_edges], dim=1)
            edge_attr = torch.cat([edge_attr, additional_edge_attr], dim=0)

        # Check for symmetry in the gap junction adjacency matrix (electrical synapses should be symmetric)
        gap_junctions = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr[:, 0]).squeeze(0)
        if not torch.allclose(gap_junctions.T, gap_junctions):
            raise AssertionError("The gap junction adjacency matrix is not symmetric.")

        # Create the graph data object with all the processed information
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.pos = pos  # Add positional information to the graph object

        return graph, node_type, node_label, node_index, node_class, num_classes

    def save_graph_tensors(
        self,
        save_as: str,
        graph: Data,
        node_type: dict,
        node_label: dict,
        node_index: torch.Tensor,
        node_class: dict,
        num_classes: int,
    ):
        """
        Saves the graph tensors and additional attributes to a file.

        Args:
            save_as (str): Filename for the saved graph data.
            graph (Data): Processed graph data containing node features (`x`), edge indices (`edge_index`), edge attributes (`edge_attr`), node positions (`pos`), and optional node labels (`y`).
            node_type (dict): Maps node index to neuron type (e.g., sensory, motor).
            node_label (dict): Maps node index to neuron label (e.g., 'ADAL').
            node_index (torch.Tensor): Tensor of node indices.
            node_class (dict): Maps node index to neuron class (e.g., 'ADA').
            num_classes (int): Number of unique neuron types/classes.

        The graph tensors dictionary includes connectivity (`edge_index`), attributes (`edge_attr`), neuron positions (`pos`), features (`x`), and additional information such as node labels and types.
        """

        # Collect the graph data and additional attributes in a dictionary
        graph_tensors = {
            "edge_index": graph.edge_index,
            "edge_attr": graph.edge_attr,
            "pos": graph.pos,
            "x": graph.x,
            "y": graph.y,
            "node_type": node_type,
            "node_label": node_label,
            "node_class": node_class,
            "node_index": node_index,
            "num_classes": num_classes,
        }

        # Save the graph tensors to a file
        torch.save(
            graph_tensors, os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as)
        )


class DefaultPreprocessor(ConnectomeBasePreprocessor):
    """
    Default preprocessor for connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the default connectome data. It includes methods for loading, processing,
    and saving the connectome data.

    The default connectome data used here is a MATLAB preprocessed version of Cook et al., 2019 by
    Kamal Premaratne. If the raw data isn't found, please download the zip file from this link:
    https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip,
    unzip the archive in the data/raw folder, then run the MATLAB script `export_nodes_edges.m`.

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
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # # Override DefaultProprecessor with Witvliet2020Preprocessor7, a more up-to-date connectome of C. elegans.
        # return Witvliet2020Preprocessor7.preprocess(self, save_as="graph_tensors.pt")
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class ChklovskiiPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Chklovskii connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Chklovskii connectome data from the 'NeuronConnect.csv' sheet.
    It includes methods for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_chklkovskii.pt"):
            Preprocesses the Chklovskii connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_chklovskii.pt"):
        """
        Preprocesses the Chklovskii et al connectome data and saves the graph tensors to a file.

        The data is read from the XLS file named "Chklovskii_NeuronConnect.xls", which is a renaming of
        the file downloaded from https://www.wormatlas.org/images/NeuronConnect.xls. The connectome table
        is in the 'NeuronConnect.csv' sheet.

        NOTE: Description of this data from https://wormwiring.org/:
        Adult hermaphrodite, Data of Chen, Hall, and Chklovskii, 2006, Wiring optimization can relate neuronal structure and function, PNAS 103: 4723-4728 (doi:10.1073/pnas.0506806103)
        and Varshney, Chen, Paniaqua, Hall and Chklovskii, 2011, Structural properties of the C. elegans neuronal network, PLoS Comput. Biol. 3:7:e1001066 (doi:10.1371/journal.pcbi.1001066).
        Data of White et al., 1986, with additional connectivity in the ventral cord from reannotation of original electron micrographs.
        Connectivity table available through WormAtlas.org: Connectivity Data-download [.xls]
        Number of chemical and gap junction (electrical) synapses for all neurons and motor neurons. Number of NMJ’s for all ventral cord motor neurons.

        For chemical synapses, only entries with type "Sp" (send reannotated) are considered to
        avoid redundant connections.

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_cklkovskii.pt".

        Steps:
            1. Load the connectome data from the 'NeuronConnect.csv' sheet in "Chklovskii_NeuronConnect.xls".
            2. Only consider rows with "Sp" (chemical) and "EJ" (gap junction) types.
            3. Append edges and attributes (synapse strength).
            4. Ensure symmetry for electrical synapses.
            5. Convert edge attributes and edge indices to tensors.
            6. Call the `preprocess_common_tasks` method to create graph tensors.
            7. Save the graph tensors to the specified file.
        """
        # Load the XLS file and extract data from the 'NeuronConnect.csv' sheet
        df = pd.read_excel(
            os.path.join(RAW_DATA_DIR, "Chklovskii_NeuronConnect.xls"),
            sheet_name="NeuronConnect.csv",
        )

        edges = []
        edge_attr = []

        # Iterate over each row in the DataFrame
        for i in range(len(df)):
            neuron1 = df.loc[i, "Neuron 1"]  # Pre-synaptic neuron
            neuron2 = df.loc[i, "Neuron 2"]  # Post-synaptic neuron
            synapse_type = df.loc[i, "Type"]  # Synapse type (e.g., EJ, Sp)
            num_connections = df.loc[i, "Nbr"]  # Number of connections

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                if synapse_type == "EJ":  # Electrical synapse (Gap Junction)
                    edges.append([neuron1, neuron2])
                    edge_attr.append(
                        [num_connections, 0]
                    )  # Electrical synapse encoded as [num_connections, 0]

                    # Ensure symmetry by adding reverse direction for electrical synapses
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])

                elif synapse_type == "Sp":  # Only process "Sp" type chemical synapses
                    edges.append([neuron1, neuron2])
                    edge_attr.append(
                        [0, num_connections]
                    )  # Chemical synapse encoded as [0, num_connections]

        # Convert edge attributes and edge indices to torch tensors
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class OpenWormPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the OpenWorm connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the OpenWorm connectome data. It includes methods for loading, processing,
    and saving the connectome data directly from the xls file.

    Methods:
        preprocess(save_as="graph_tensors_openworm.pt"):
            Preprocesses the OpenWorm connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_openworm.pt"):
        """
        Preprocesses the OpenWorm connectome data and saves the graph tensors to a file.

        The data is read directly from an XLS file named "OpenWorm_CElegansNeuronTables.xls", which is a rename of the
        file downloaded from the OpenWorm repository: https://github.com/openworm/c302/blob/master/c302/CElegansNeuronTables.xls

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_openworm.pt".

        Steps:
            1. Load the connectome data from the first sheet of the "OpenWorm_CElegansNeuronTables.xls" file.
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        # Load the XLS file and extract data from the first sheet "Connectome"
        df = pd.read_excel(
            os.path.join(RAW_DATA_DIR, "OpenWorm_CElegansNeuronTables.xls"), sheet_name="Connectome"
        )

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "Origin"]
            neuron2 = df.loc[i, "Target"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # Determine the synapse type and number of connections
                synapse_type = df.loc[i, "Type"]
                num_connections = df.loc[i, "Number of Connections"]

                # Add the connection between neuron1 and neuron2
                edges.append([neuron1, neuron2])

                if synapse_type == "GapJunction":  # electrical synapse
                    edge_attr.append(
                        [num_connections, 0]
                    )  # electrical synapse encoded as [num_connections, 0]

                    # Ensure symmetry for gap junctions by adding reverse connection
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif synapse_type == "Send":  # chemical synapse
                    edge_attr.append(
                        [0, num_connections]
                    )  # chemical synapse encoded as [0, num_connections]

        # Convert to torch tensors
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        The data is read from an Excel file named "CElegansFunctionalConnectivity.xlsx" which is a renaming of the
        Supplementary Table 1 file "1586_2023_6683_MOESM3_ESM.xlsx" downloaded from the Supplementary information of the paper
        "Randi, F., Sharma, A. K., Dvali, S., & Leifer, A. M. (2023). Neural signal propagation atlas of Caenorhabditis elegans. Nature, 623(7986), 406–414. https://doi.org/10.1038/s41586-023-06683-4"
        at this direct link: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06683-4/MediaObjects/41586_2023_6683_MOESM3_ESM.xlsx

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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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
    Interpolate data using scipy's interp1d or np.interp.

    This function interpolates the given data to the desired time intervals.

    Parameters:
        time (numpy.ndarray): 1D array containing the time points corresponding to the data.
        data (numpy.ndarray): A 2D array containing the data to be interpolated, with shape (time, neurons).
        target_dt (float): The desired time interval between the interpolated data points.
        method (str, optional): The interpolation method to use. Default is 'linear'.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the interpolated time points and data.
    """
    # Check if correct interpolation method provided
    assert method in {
        None,
        "linear",
        "quadratic",
        "cubic",
    }, "Invalid interpolation method. Choose from [None, 'linear', 'cubic', 'quadratic']."
    assert time.shape[0] == data.shape[0], "Input temporal dimension mismatch."
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data
    # Ensure that time is a 1D array
    time = time.squeeze()
    # Create the target time vector, ensuring the range does not exceed the original data range
    target_time_np = np.arange(time.min(), time.max(), target_dt)
    num_neurons = data.shape[1]
    interpolated_data_np = np.zeros((len(target_time_np), num_neurons), dtype=np.float32)
    # Use scipy's interpolation method
    # TODO: Vectorize this operation.
    if method == "linear":
        for i in range(num_neurons):
            interpolated_data_np[:, i] = np.interp(target_time_np, time, data[:, i])
    else:
        logger.info(
            "Warning: scipy.interplate.interp1d is deprecated. Best to choose method='linear'."
        )
        for i in range(num_neurons):
            interp = interp1d(
                x=time, y=data[:, i], kind=method, bounds_error=False, fill_value="extrapolate"
            )
            interpolated_data_np[:, i] = interp(target_time_np)
    # Reshape interpolated time vector to (time, 1)
    target_time_np = target_time_np.reshape(-1, 1)
    # Final check for shape consistency
    assert target_time_np.shape[0] == interpolated_data_np.shape[0], "Output temporal dimension."
    # Return the interpolated time and data
    return target_time_np, interpolated_data_np


def aggregate_data(time, data, target_dt):
    """
    Downsample data using aggregation.

    This function downsamples the data by averaging over intervals of size `target_dt`.

    Parameters:
        time (numpy.ndarray): 1D array containing the time points corresponding to the data.
        data (numpy.ndarray): A 2D array containing the data to be downsampled, with shape (time, neurons).
        target_dt (float): The desired time interval between the downsampled data points.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the downsampled time points and data.
    """
    assert time.shape[0] == data.shape[0], "Input temporal dimension."
    # If target_dt is None, return the original data
    if target_dt is None:
        return time, data
    # Ensure that time is a 1D array
    time = time.squeeze()
    # Compute the downsample rate
    original_dt = np.median(np.diff(time, axis=0)[1:]).item()
    interval_width = max(1, int(np.round(target_dt / original_dt)))
    num_intervals = len(time) // interval_width
    # Create the downsampled time array
    target_time_np = target_dt * np.arange(num_intervals)
    # Create the downsampled data array
    num_neurons = data.shape[1]
    downsampled_data = np.zeros((num_intervals, num_neurons), dtype=np.float32)
    # Downsample the data by averaging over intervals
    # TODO: Vectorize this operation.
    for i in range(num_neurons):
        reshaped_data = data[: num_intervals * interval_width, i].reshape(
            num_intervals, interval_width
        )
        downsampled_data[:, i] = reshaped_data.mean(axis=1)
    # Reshape downsampled time vector to (time, 1)
    target_time_np = target_time_np.reshape(-1, 1)
    # Final check for shape consistency
    assert (
        target_time_np.shape[0] == downsampled_data.shape[0]
    ), "Output temporal dimension mismatch."
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
        labeled_neuron_to_idx (dict): Mapping of labeled neurons to their indices.
        unlabeled_neuron_to_idx (dict): Mapping of unlabeled neurons to their indices.
        slot_to_labeled_neuron (dict): Mapping of slots to labeled neurons.
        slot_to_unlabeled_neuron (dict): Mapping of slots to unlabeled neurons.
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
        _fill_labeled_neurons_data():
            Fills data for labeled neurons.
        _fill_calcium_data(idx, slot):
            Fills calcium data for a given neuron index and slot.
        _fill_unlabeled_neurons_data():
            Fills data for unlabeled neurons.
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
        self.labeled_neuron_to_idx = dict()
        self.unlabeled_neuron_to_idx = dict()
        self.slot_to_labeled_neuron = dict()
        self.slot_to_unlabeled_neuron = dict()
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
        self._fill_labeled_neurons_data()
        self._fill_unlabeled_neurons_data()
        self._update_worm_dataset()
        self._remove_old_mappings()

    def _prepare_initial_data(self):
        """
        Prepares initial data structures for reshaping.
        """
        assert (
            len(self.idx_to_neuron) == self.calcium_data.shape[1]
        ), "Number of neurons in calcium data matrix does not match number of recorded neurons."
        self.labeled_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        self.unlabeled_neurons_mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
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

    def _fill_labeled_neurons_data(self):
        """
        Fills data for labeled neurons.
        """
        for slot, neuron in enumerate(NEURON_LABELS):
            if neuron in self.neuron_to_idx:  # labeled neuron
                idx = self.neuron_to_idx[neuron]
                self.labeled_neuron_to_idx[neuron] = idx
                self._fill_calcium_data(idx, slot)
                self.labeled_neurons_mask[slot] = True
                self.slot_to_labeled_neuron[slot] = neuron

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

    def _fill_unlabeled_neurons_data(self):
        """
        Fills data for unlabeled neurons.
        """
        free_slots = list(np.where(~self.labeled_neurons_mask)[0])
        for neuron in set(self.neuron_to_idx) - set(self.labeled_neuron_to_idx):
            self.unlabeled_neuron_to_idx[neuron] = self.neuron_to_idx[neuron]
            slot = np.random.choice(free_slots)
            free_slots.remove(slot)
            self.slot_to_unlabeled_neuron[slot] = neuron
            self._fill_calcium_data(self.neuron_to_idx[neuron], slot)
            self.unlabeled_neurons_mask[slot] = True

    def _update_worm_dataset(self):
        """
        Updates the worm dataset with reshaped data and mappings.
        """
        self.slot_to_neuron.update(self.slot_to_labeled_neuron)
        self.slot_to_neuron.update(self.slot_to_unlabeled_neuron)
        self.worm_dataset.update(
            {
                "calcium_data": self.standard_calcium_data,  # normalized, resampled
                "dt": self.dt,  # resampled (vector)
                "idx_to_labeled_neuron": {v: k for k, v in self.labeled_neuron_to_idx.items()},
                "idx_to_unlabeled_neuron": {v: k for k, v in self.unlabeled_neuron_to_idx.items()},
                "median_dt": self.median_dt,  # resampled (scalar)
                "labeled_neuron_to_idx": self.labeled_neuron_to_idx,
                "labeled_neuron_to_slot": {v: k for k, v in self.slot_to_labeled_neuron.items()},
                "labeled_neurons_mask": self.labeled_neurons_mask,
                "neuron_to_slot": {v: k for k, v in self.slot_to_neuron.items()},
                "neurons_mask": self.labeled_neurons_mask | self.unlabeled_neurons_mask,
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
                "slot_to_labeled_neuron": self.slot_to_labeled_neuron,
                "slot_to_neuron": self.slot_to_neuron,
                "slot_to_unlabeled_neuron": self.slot_to_unlabeled_neuron,
                "time_in_seconds": self.time_in_seconds,  # resampled
                "unlabeled_neuron_to_idx": self.unlabeled_neuron_to_idx,
                "unlabeled_neuron_to_slot": {
                    v: k for k, v in self.slot_to_unlabeled_neuron.items()
                },
                "unlabeled_neurons_mask": self.unlabeled_neurons_mask,
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
        source_dataset,
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
            source_dataset (str): The name of the source dataset to be preprocessed.
            transform (object, optional): The sklearn transformation to be applied to the data. Default is StandardScaler().
            smooth_method (str, optional): The smoothing method to apply to the data. Default is "moving".
            interpolate_method (str, optional): The interpolation method to use when resampling the data. Default is "linear".
            resample_dt (float, optional): The resampling time interval in seconds. Default is 0.1.
            **kwargs: Additional keyword arguments for smoothing.
        """
        self.source_dataset = source_dataset
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

    def resample_data(self, time_in_seconds, ca_data, upsample=True):
        """
        Resample the calcium data to the desired time steps.
        The input time vector and data matrix should be matched in time,
        and the resampled time vector and data matrix should also be matched.

        Parameters:
            time_in_seconds (np.ndarray): Time vector in seconds with shape (time, 1).
            ca_data (np.ndarray): Original, non-uniformly sampled calcium data with shape (time, neurons).
            upsample (bool, optional): Whether to sample at a higher frequency (i.e., with smaller dt). Default is True.

        Returns:
            np.ndarray, np.ndarray: Resampled time vector and calcium data.
        """
        assert time_in_seconds.shape[0] == ca_data.shape[0], (
            f"Input mismatch! Time vector length ({time_in_seconds.shape[0]}) "
            f"doesn't match data length ({ca_data.shape[0]})."
        )
        # Perform upsampling (interpolation) or downsampling (aggregation) as needed
        if upsample:
            interp_time, interp_ca = interpolate_data(
                time_in_seconds,
                ca_data,
                target_dt=self.resample_dt,
                method=self.interpolate_method,
            )
        else:
            # First upsample to a finer dt before downsampling
            interp_time, interp_ca = interpolate_data(
                time_in_seconds,
                ca_data,
                target_dt=self.resample_dt / 6,  # Finer granularity first
                method=self.interpolate_method,
            )
            # Then aggregate over intervals to match the desired dt
            interp_time, interp_ca = aggregate_data(
                interp_time,
                interp_ca,
                target_dt=self.resample_dt,
            )
        # Ensure the resampled time and data are the same shape
        if interp_time.shape[0] != interp_ca.shape[0]:
            raise ValueError(
                f"Resampling mismatch! Resampled time vector ({interp_time.shape[0]}) "
                f"doesn't match resampled data length ({interp_ca.shape[0]})."
            )
        return interp_time, interp_ca

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
            int: Number of labeled neurons.
        """
        # TODO: Supplement this this with the Leifer2023 version so that we only need this one definition.
        idx_to_neuron = {
            nid: (
                str(nid)
                if (j is None or isinstance(j, np.ndarray) or j == "merge" or not j.isalnum())
                else str(j)
            )
            for nid, j in enumerate(unique_IDs)
        }
        idx_to_neuron = {
            nid: (
                name.replace("0", "") if not name.endswith("0") and not name.isnumeric() else name
            )
            for nid, name in idx_to_neuron.items()
        }
        idx_to_neuron = {
            nid: (str(nid) if name not in set(NEURON_LABELS) else name)
            for nid, name in idx_to_neuron.items()
        }
        neuron_to_idx = dict((v, k) for k, v in idx_to_neuron.items())
        num_labeled_neurons = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of labled neurons
        return neuron_to_idx, num_labeled_neurons

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

    def get_longest_nan_stretch(self, arr):
        """
        Calculate the longest continuous stretch of NaNs in a 1D array.

        Parameters:
        arr (np.array): 1D Input array to check.

        Returns:
        int: Length of the longest continuous stretch of NaNs.
        """
        assert arr.ndim == 1, "Array must be a 1D time series."
        isnan = np.isnan(arr)
        if not np.any(isnan):
            return 0
        stretches = np.diff(
            np.where(np.concatenate(([isnan[0]], isnan[:-1] != isnan[1:], [True])))[0]
        )[::2]
        return stretches.max() if len(stretches) > 0 else 0

    def filter_bad_traces_by_nan_stretch(self, data, nan_stretch_threshold=0.05):
        """
        Filters out traces with long stretches of NaNs.

        Parameters:
        data (np.array): The neural data array with shape (time_points, neurons).
        nan_stretch_threshold (float): Proportion of the total recording time above which traces are considered bad.

        Returns:
        (np.array, np.array): Tuple of filtered neural data and the associated mask into the original data array.
        """
        t, n = data.shape
        max_nan_stretch_allowed = int(t * nan_stretch_threshold)
        bad_traces_mask = (
            np.apply_along_axis(self.get_longest_nan_stretch, 0, data) > max_nan_stretch_allowed
        )
        good_traces_mask = ~bad_traces_mask
        filtered_data = data[:, good_traces_mask]
        return filtered_data, good_traces_mask

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

        There are weird-looking traces in some raw data caused by interpolations of missing values
        (NaNs) when neurons were not consistently tracked over time due to imperfect nonrigid registration.
        This helper function was written to filter out these problematic imputed neural traces.

        Parameters:
        data (np.array): The neural data array with shape (time_points, neurons).
        window_size (int): The size of the window to check for linearity.
        linear_segment_threshold (float): Proportion of linear segments above which traces are considered bad.

        Returns:
        (np.array, nparray): Tuple of filtered neural data and the associated mask into the original data array.
        """
        t, n = data.shape
        linear_segments = np.zeros(n, dtype=int)
        window_start = range(
            0, t - window_size, window_size // 2
        )  # overlapping/staggered windows faster than non-overlapping
        for i in window_start:
            segment = data[i : i + window_size, :]
            ls = np.apply_along_axis(self.is_monotonic_linear, 0, segment)
            linear_segments += ls.astype(int)
        proportion_linear = linear_segments / len(window_start)
        bad_traces_mask = np.array(proportion_linear > linear_segment_threshold)
        good_traces_mask = ~bad_traces_mask
        filtered_data = data[:, good_traces_mask]
        return filtered_data, good_traces_mask

    def load_data(self, file_name):
        """
        Load the raw data from a .mat file.
        The  simple place-holder implementation seen here for the
        Skora, Kato, Nichols, Uzel, and Kaplan datasets but should
        be customized for the others.

        Parameters:
            file_name (str): The name of the file to load.

        Returns:
            dict: The loaded data.
        """
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))

    def extract_data(self):
        """
        Extract the basic data (neuron IDs, calcium traces, and time vector) from the raw data file.
        This method should be overridden by subclasses to implement dataset-specific extraction logic.
        """
        raise NotImplementedError()

    def create_metadata(self, **kwargs):
        """
        Create a dictionary of extra information or metadata for a dataset.

        Returns:
            dict: A dictionary of extra information or metadata.
        """
        extra_info = dict()
        extra_info.update(kwargs)
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
        metadata=dict(),
    ):
        """
        Helper function for preprocessing calcium fluorescence neural data from one worm.
        This method checks that the neuron labels, data matrix and time vector are of consistent
        shapes (e.g. number of timesteps in data matrix should be same as length of time vector).
        Any empty data (e.g. no labeled neurons or no recorded activity data) are thrown out.

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
            Iterate through the traces and preprocess each one:
                1. Normalize the calcium data.
                2. Compute the residual calcium.
                3. Smooth the data.
                4. Resample the data.
                5. Name the worm and update the index.
            Save the resulting data.
        """
        assert (
            len(neuron_IDs) == len(traces) == len(raw_timeVectorSeconds)
        ), "Lists for neuron labels, activity data, and time vectors must all be the same length."
        # Each worm has a unique set of neurons, time vectors and calcium traces
        for i, trace_data in enumerate(traces):
            # Matrix `trace_data` should be shaped as (time, neurons)
            assert trace_data.ndim == 2, "Calcium traces must be 2D arrays."
            assert trace_data.shape[0] == len(
                raw_timeVectorSeconds[i]
            ), "Calcium trace does not have the right number of time points."
            assert trace_data.shape[1] == len(
                neuron_IDs[i]
            ), "Calcium trace does not have the right number of neurons."
            # Ignore any worms with empty traces
            if trace_data.size == 0:
                continue
            # Ignore any worms with very short recordings
            if len(raw_timeVectorSeconds[i]) < 600:
                continue
            # Map labeled neurons
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
            neuron_to_idx, num_labeled_neurons = self.create_neuron_idx(unique_IDs)
            # Ignore any worms with no labelled neurons
            if num_labeled_neurons == 0:
                continue
            # Only get data for unique neurons
            trace_data = trace_data[:, unique_indices.astype(int)]
            # Normalize calcium data
            calcium_data = self.normalize_data(trace_data)  # matrix
            # Compute residual calcium
            time_in_seconds = raw_timeVectorSeconds[i].reshape(raw_timeVectorSeconds[i].shape[0], 1)
            time_in_seconds = np.array(time_in_seconds, dtype=np.float32)  # vector
            time_in_seconds = time_in_seconds - time_in_seconds[0]  # start at 0.0 seconds
            dt = np.diff(time_in_seconds, axis=0, prepend=0.0)  # vector
            original_median_dt = np.median(dt[1:]).item()  # scalar
            residual_calcium = np.gradient(
                calcium_data, time_in_seconds.squeeze(), axis=0
            )  # vector
            # Smooth data
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(residual_calcium, time_in_seconds)
            # Resample data
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
            num_unlabeled_neurons = int(num_neurons) - num_labeled_neurons
            # Name worm and update index
            worm = "worm" + str(worm_idx)  # use global worm index
            worm_idx += 1  # increment worm index
            # Save data
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
                    "num_labeled_neurons": num_labeled_neurons,
                    "num_neurons": int(num_neurons),
                    "num_unlabeled_neurons": num_unlabeled_neurons,
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
                    "extra_info": self.create_metadata(
                        **metadata
                    ),  # additional information and metadata
                }
            }
            # Update preprocessed data collection
            preprocessed_data.update(worm_dict)
        # Return the updated preprocessed data and worm index
        return preprocessed_data, worm_idx


class Kato2015Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Kato et al., 2015 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Kato et al., 2015 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
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
        self.citation = "Kato, S., Kaplan, H. S., Schrödel, T., Skora, S., Lindsay, T. H., Yemini, E., Lockery, S., & Zimmer, M. (2015). Global brain dynamics embed the motor command sequence of Caenorhabditis elegans. Cell, 163(3), 656–669. https://doi.org/10.1016/j.cell.2015.09.034"

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
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
        self.citation = "Nichols, A. L. A., Eichler, T., Latham, R., & Zimmer, M. (2017). A global brain state underlies C. elegans sleep behavior. Science, 356(6344). https://doi.org/10.1126/science.aam6851"

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
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
        self.citation = "Skora, S., Mende, F., & Zimmer, M. (2018). Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans. Cell Reports, 22(4), 953–966. https://doi.org/10.1016/j.celrep.2017.12.091"

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
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
        self.citation = "Kaplan, H. S., Salazar Thula, O., Khoss, N., & Zimmer, M. (2020). Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales. Neuron, 105(3), 562-576.e9. https://doi.org/10.1016/j.neuron.2019.10.037"

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Nejatbakhsh2020Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Nejatbakhsh et al., 2020 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Nejatbakhsh et al., 2020 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(file): Extracts neuron IDs, calcium traces, and time vector from the NWB file.
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
        self.citation = "Nejatbakhsh, A., Varol, E., Yemini, E., Venkatachalam, V., Lin, A., Samuel, A. D. T., & Paninski, L. (2020). Extracting neural signals from semi-immobilized animals with deformable non-negative matrix factorization. In bioRxiv (p. 2020.07.07.192120). https://doi.org/10.1101/2020.07.07.192120"

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
            # TODO: Impute missing NaN values.
            neuron_ids = np.array(
                read_nwbfile.processing["CalciumActivity"].data_interfaces["NeuronIDs"].labels,
                dtype=np.dtype(str),
            )
            # sampling frequency is 4 Hz
            time_vector = np.arange(0, traces.shape[0]).astype(np.dtype(float)) / 4
        # Return the extracted data
        return neuron_ids, traces, time_vector

    def preprocess(self):
        """
        Preprocesses the Nejatbakhsh et al., 2020 neural data and saves it as a pickle file

        The data is read from NWB files located in subdirectories nested in this source dataset's directory.

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
            # Skip hidden/system files like .DS_Store
            if tree.startswith("."):
                continue
            subfolder = os.path.join(self.raw_data_path, self.source_dataset, tree)
            if not os.path.isdir(subfolder):
                continue
            for file_name in os.listdir(subfolder):
                # Ignore non-NWB files
                if not file_name.endswith(".nwb"):
                    continue
                neuron_ids, traces, raw_time_vector = self.extract_data(
                    os.path.join(self.raw_data_path, self.source_dataset, subfolder, file_name)
                )
                metadata = dict(
                    citation=self.citation,
                    data_file=os.path.join(
                        os.path.basename(self.raw_data_path), self.source_dataset, file_name
                    ),
                )
                preprocessed_data, worm_idx = self.preprocess_traces(
                    [neuron_ids],
                    [traces],
                    [raw_time_vector],
                    preprocessed_data,
                    worm_idx,
                    metadata,
                )
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
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
        self.citation = "Yemini, E., Lin, A., Nejatbakhsh, A., Varol, E., Sun, R., Mena, G. E., Samuel, A. D. T., Paninski, L., Venkatachalam, V., & Hobert, O. (2021). NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans. Cell, 184(1), 272-288.e11. https://doi.org/10.1016/j.cell.2020.12.012"

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
            # Observed empirically that the first three values of activity equal 0.0s
            activity = activity[4:]
            tvec = tvec[4:]
            # Impute any remaining NaN values
            # NOTE: This is very slow with the default settings!
            imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
            if np.isnan(activity).any():
                activity = imputer.fit_transform(activity)
            # Add activity to list of traces
            traces.append(activity)
            # Add time vector to list of time vectors
            time_vector_seconds.append(tvec)
        # Return the extracted data
        return neuron_IDs, traces, time_vector_seconds

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
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
        preprocess(): Preprocesses the Uzel et al., 2022 neural data and saves is as a file.
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
        self.citation = "Uzel, K., Kato, S., & Zimmer, M. (2022). A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans. Current Biology: CB, 32(16), 3443-3459.e8. https://doi.org/10.1016/j.cub.2022.06.039"

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
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
        preprocess(): Preprocesses the Dag et al., 2023 neural data and saves it as a pickle file.
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
        self.citation = "Dag, U., Nwabudike, I., Kang, D., Gomes, M. A., Kim, J., Atanas, A. A., Bueno, E., Estrem, C., Pugliese, S., Wang, Z., Towlson, E., & Flavell, S. W. (2023). Dissecting the functional organization of the C. elegans serotonergic system at whole-brain scale. Cell, 186(12), 2574-2592.e20. https://doi.org/10.1016/j.cell.2023.04.023"

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
            # Indices in index_map correspond to labeled neurons
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
            # Remaining indices correspond to unlabeled neurons
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
        # Neurons with DV/LR ambiguity have '?' or '??' in labels that must be inferred
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
        # Make the extracted data into a list of arrays
        all_IDs = [neurons_copy]
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
        noid_data_files = os.path.join(self.raw_data_path, "Dag2023", "swf415_no_id")  # unused
        # 'NeuroPAL_labels_dict.json' maps data file names to a dictionary of neuron label information
        labels_file = "NeuroPAL_labels_dict.json"
        # First deal with the swf702_with_id which contains data from labeled neurons
        for file_name in os.listdir(withid_data_files):
            if not file_name.endswith(".h5"):
                continue
            data_file = os.path.join("swf702_with_id", file_name)
            neurons, raw_traces, time_vector_seconds = self.extract_data(data_file, labels_file)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                raw_traces,
                time_vector_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Next deal with the swf415_no_id which contains purely unlabeled neuron data
        # NOTE: These won't get used at all as they are skipped in NeuralBasePreprocessor.preprocess_traces since num_labeled_neurons is 0.
        for file_name in os.listdir(noid_data_files):
            if not file_name.endswith(".h5"):
                continue
            data_file = os.path.join("swf415_no_id", file_name)
            neurons, raw_traces, time_vector_seconds = self.extract_data(data_file, labels_file)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                raw_traces,
                time_vector_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")
        return None


class Flavell2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Atanas et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Atanas et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        find_nearest_label(query, possible_labels, char="?"): Finds the nearest neuron label from a list given a query.
        load_data(file_name): Loads the data from the specified HDF5 or JSON file.
        extract_data(file_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        preprocess(): Preprocesses the Flavell et al., 2023 neural data and saves it as pickle file.
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
        self.citation = "Atanas, A. A., Kim, J., Wang, Z., Bueno, E., Becker, M., Kang, D., Park, J., Kramer, T. S., Wan, F. K., Baskoylu, S., Dag, U., Kalogeropoulou, E., Gomes, M. A., Estrem, C., Cohen, N., Mansinghka, V. K., & Flavell, S. W. (2023). Brain-wide representations of behavior spanning multiple timescales and states in C. elegans. Cell. https://doi.org/10.1016/j.cell.2023.07.035"

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
        # Neurons with DV/LR ambiguity have '?' or '??' in labels that must be inferred
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
        # Make the extracted data into a list of arrays
        all_IDs = [neurons_copy]
        all_traces = [calcium_data]
        timeVectorSeconds = [time_in_seconds]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Atanas et al., 2023 neural data and saves it as a pickle file.

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
        # Load and preprocess data
        preprocessed_data = dict()
        worm_idx = 0
        for file_name in os.listdir(os.path.join(self.raw_data_path, self.source_dataset)):
            if not (file_name.endswith(".h5") or file_name.endswith(".json")):
                continue
            file_data = self.load_data(file_name)  # load
            neurons, calcium_data, time_in_seconds = self.extract_data(file_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                calcium_data,
                time_in_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Leifer2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Randi et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Leifer et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        str_to_float(str_num): Converts a string in scientific notation to a floating-point number.
        load_labels(file_path): Loads neuron labels from a text file.
        load_time_vector(file_path): Loads the time vector from a text file.
        load_data(file_path): Loads the neural data from a text file.
        create_neuron_idx(label_list): Creates a mapping of neuron labels to indices.
        extract_data(data_file, labels_file, time_file): Extracts neuron IDs, calcium traces, and time vector from the loaded data files.
        preprocess(): Preprocesses the Leifer et al., 2023 neural data and saves it as a pickle a file.
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
        self.citation = "Randi, F., Sharma, A. K., Dvali, S., & Leifer, A. M. (2023). Neural signal propagation atlas of Caenorhabditis elegans. Nature, 623(7986), 406–414. https://doi.org/10.1038/s41586-023-06683-4"

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

    def create_neuron_idx(self, label_list):
        """
        Creates a mapping of neuron labels to indices.

        Parameters:
            label_list (list): The list of neuron labels.

        Returns:
            tuple: A tuple containing the neuron-to-index mapping and the number of labeled neurons.
        """
        neuron_to_idx = dict()
        num_unlabeled_neurons = 0
        for j, item in enumerate(label_list):
            previous_list = label_list[:j]
            if not item.isalnum():  # happens when the label is empty string ''
                label_list[j] = str(j)
                num_unlabeled_neurons += 1
                neuron_to_idx[str(j)] = j
            else:
                if item in NEURON_LABELS and item not in previous_list:
                    neuron_to_idx[item] = j
                # If a neuron label repeated assume a mistake and treat the duplicate as an unlabeled neuron
                elif item in NEURON_LABELS and item in previous_list:
                    label_list[j] = str(j)
                    num_unlabeled_neurons += 1
                    neuron_to_idx[str(j)] = j
                # Handle ambiguous neuron labels
                else:
                    if str(item + "L") in NEURON_LABELS and str(item + "L") not in previous_list:
                        label_list[j] = str(item + "L")
                        neuron_to_idx[str(item + "L")] = j
                    elif str(item + "R") in NEURON_LABELS and str(item + "R") not in previous_list:
                        label_list[j] = str(item + "R")
                        neuron_to_idx[str(item + "R")] = j
                    else:  # happens when the label is "merge"; TODO: Ask authors what that is?
                        label_list[j] = str(j)
                        num_unlabeled_neurons += 1
                        neuron_to_idx[str(j)] = j
        num_labeled_neurons = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of labeled neurons
        assert (
            num_labeled_neurons == len(label_list) - num_unlabeled_neurons
        ), "Incorrect calculation of the number of labeled neurons."
        return neuron_to_idx, num_labeled_neurons

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
        # Remove neurons with long stretches of NaNs
        real_data, nan_mask = self.filter_bad_traces_by_nan_stretch(real_data)
        label_list = np.array(label_list, dtype=str)[nan_mask].tolist()
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(real_data).any():
            real_data = imputer.fit_transform(real_data)
        # Remove badly imputed neurons from the data
        filt_real_data, filt_mask = self.filter_bad_traces_by_linear_segments(real_data)
        filt_label_list = np.array(label_list, dtype=str)[filt_mask].tolist()
        # Make the extracted data into a list of arrays
        all_IDs = [filt_label_list]
        all_traces = [filt_real_data]
        timeVectorSeconds = [time_in_seconds]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Randi et al., 2023 neural data and saves it as a pickle file.

        The data is read from text files located in the dataset directory.

        NOTE: The `preprocess` method for the Leifer 2023 dataset is significantly different
            than that for the other datasets due to differences between the file structure containing
            the raw data for the Leifer2023 dataset compared to the other source datasets:
                - Leifer2023 raw data uses 6 files per worm each containing distinct information.
                - The other datasets use 1 file containing all the information for multiple worms.
            Unlike the `preprocess` method in the other dataset classes which makes use of the
            `preprocess_traces` method from the parent NeuralBasePreprocessor class, this one does not.

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
        # Load and preprocess data
        preprocessed_data = dict()
        data_dir = os.path.join(self.raw_data_path, self.source_dataset)
        # Every worm has 6 text files
        files = os.listdir(data_dir)
        num_worms = int(len(files) / 6)
        # Initialize worm index outside file loop
        worm_idx = 0
        # Iterate over each worm's triad of data text files
        for i in tqdm(range(0, num_worms)):
            data_file = os.path.join(data_dir, f"{str(i)}_gcamp.txt")
            labels_file = os.path.join(data_dir, f"{str(i)}_labels.txt")
            time_file = os.path.join(data_dir, f"{str(i)}_t.txt")
            # Load and extract raw data
            label_list, real_data, time_in_seconds = self.extract_data(
                data_file, labels_file, time_file
            )  # extract
            file_name = str(i) + "_{gcamp|labels|t}.txt"
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            # Preprocess raw data
            preprocessed_data, worm_idx = self.preprocess_traces(
                label_list,
                real_data,
                time_in_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
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
        self.citation = "Lin, A., Qin, S., Casademunt, H., Wu, M., Hung, W., Cain, G., Tan, N. Z., Valenzuela, R., Lesanpezeshki, L., Venkatachalam, V., Pehlevan, C., Zhen, M., & Samuel, A. D. T. (2023). Functional imaging and quantification of multineuronal olfactory responses in C. elegans. Science Advances, 9(9), eade1249. https://doi.org/10.1126/sciadv.ade1249"

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
        # Filter for proofread neurons
        _filter = dataset_raw["use_flag"].flatten() > 0
        neurons = [str(_.item()) for _ in dataset_raw["proofread_neurons"].flatten()[_filter]]
        raw_time_vec = np.array(dataset_raw["times"].flatten()[0][-1])
        raw_activitiy = dataset_raw["corrected_F"][_filter].T  # (time, neurons)
        # Replace first nan with F0 value
        _f0 = dataset_raw["F_0"][_filter][:, 0]
        raw_activitiy[0, :] = _f0
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(raw_activitiy).any():
            raw_activitiy = imputer.fit_transform(raw_activitiy)
        # Make the extracted data into a list of arrays
        neuron_IDs, raw_traces, time_vector_seconds = [neurons], [raw_activitiy], [raw_time_vec]
        # Return the extracted data
        return neuron_IDs, raw_traces, time_vector_seconds

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
        for file_name in tqdm(os.listdir(data_files)):
            if not file_name.endswith(".mat"):
                continue
            neurons, raw_traces, time_vector_seconds = self.extract_data(file_name)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                raw_traces,
                time_vector_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


class Venkatachalam2024Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Venkatachalam (2024, unpublished) connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Venkatachalam (2024, unpublished) neural data. It includes methods for loading,
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
        self.citation = "Seyedolmohadesin, M, unpublished 2024, _Brain-wide neural activity data in C. elegans_. https://chemosensory-data.worm.world/ [Last Accessed: October 3, 2024]"

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
            zip_ref.extractall(source_directory)
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
        # 9 columns + 98 columns of blank neural data
        time_vector = (
            data.columns[107:-1].astype(float).to_numpy() * 0.375
        )  # Columns 9 onwards contain calcium data with dt of 375ms
        traces = data.iloc[:, 107:-1].values.T  # transpose to get (time, neurons)
        # Remove neuron traces that are all NaN values
        mask = np.argwhere(~np.isnan(traces).all(axis=0)).flatten()
        traces = traces[:, mask]
        # Get the neuron labels corresponding to the traces
        neuron_ids = np.array(data["neuron"].unique(), dtype=str)[mask].tolist()
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(traces).any():
            traces = imputer.fit_transform(traces)
        # Make the extracted data into a list of arrays
        all_IDs = [neuron_ids]
        all_traces = [traces]
        timeVectorSeconds = [time_vector]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

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
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_ids,
                traces,
                raw_time_vector,
                preprocessed_data,
                worm_idx,
                metadata,
            )
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        self.save_data(preprocessed_data)
        logger.info(f"Finished processing {self.source_dataset}.")


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
