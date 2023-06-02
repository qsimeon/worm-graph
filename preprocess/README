# Preprocessing Submodule

This repository contains the functions used to preprocess the open-source calcium imaging data for neural activity analysis. The datasets are preprocessed and organized in a standard manner, making them ready to be used for various tasks such as neural network training, data analysis, and visualization.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the pipeline for preprocessing the open source data as specified in the configuration file.
- `_utils.py`: Contains utility functions and classes for data processing.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file in `conf/preprocess.yaml` as desired.
3. Run the main pipeline: `python preprocess/_main.py`.
4. The processed dataset will be saved in the `data/processed/neural` folder.
5. Load the preprocessed data using our `data` submodule and have fun!

## Datasets

The following datasets are included in this submodule:

- `Kato2015`: Calcium imaging data from **12** *C. elegans* individuals.
- `Nichols2017`: Calcium imaging data from **44** *C. elegans* individuals.
- `Skora2018`: Calcium imaging data from **12** *C. elegans* individuals.
- `Kaplan2020`: Calcium imaging data from **19** *C. elegans* individuals.
- `Uzel2022`: Calcium imaging data from **6** *C. elegans* individuals.
- `Leifer2023`: Calcium imaging data from **41** *C. elegans* individuals.
- `Flavell2023`: Calcium imaging data from **10** *C. elegans* individuals.

### Dataset Structure

Each dataset is stored in a Python dictionary:

<details>
<summary>Here you will find its list of features</summary>

- `dataset`: (str) Name of the dataset
- `smooth_method`: (str) Method used to smooth the calcium data
- `worm`: (str) The worm ID in the dataset
- `max_timesteps`: (float) Number of time steps of the data
- `dt`: (torch.tensor) Column vector containing the difference between time steps. Shape (max_timesteps, 1)
- `calcium_data`: (torch.tensor) The calcium data, with standardized columns. Shape: (max_timesteps, 302)
- `smooth_calcium_data`: (torch.tensor) Smoothed calcium data, with standardized columns. Shape: (max_timesteps, 302)
- `residual_calcium`: (torch.tensor) The residual calcium data, with standardized columns. Shape: (max_timesteps, 302)
- `smooth_residual_calcium`: (torch.tensor) Smoothed residual calcium data, with standardized columns. Shape: (max_timesteps, 302)
- `time_in_seconds`: (torch.tensor) A column vector equally spaced by dt. Shape: (max_timesteps, 1)
- `num_neurons`: (int) Number of total tracked neurons of this specific worm
- `num_named_neurons`: (int) Number of labeled neurons
- `num_unknown_neurons`: (int) Number of unlabeled neurons
- `named_neurons_mask`: (torch.tensor) A bool vector indicating the positions of the labeled neurons. Shape: (302)
- `unknown_neurons_mask`: (torch.tensor) A bool vector indicating the positions of the unlabeled neurons. Shape: (302)
- `neurons_mask`: (torch.tensor) A bool vector indicating the positions of all tracked neurons (labeled + unlabeled). Shape: (302)
- `slot_to_named_neuron`: (dict) Mapping of column index -> 302 neurons. Len: num_neurons
- `named_neuron_to_slot`: (dict) Mapping of 302 neurons -> column index. Len: num_neurons
- `slot_to_unknown_neuron`: (dict) Mapping of column index -> unlabeled neuron. Len: num_unknown_neurons
- `unknown_neuron_to_slot`: (dict) Mapping of unlabeled neurons -> column index. Len: num_unknown_neurons
- `slot_to_neuron`: (dict) Mapping of column index -> labeled+unlabeled neurons. Len: num_neurons
- `neuron_to_slot`: (dict) Mapping of labeled+unlabeled neurons -> column index. Len: num_neurons

</details>

## Preprocessing

The datasets have been preprocessed using Python scripts available in this repository. The preprocessing steps include:

- Loading raw data in various formats (MATLAB files, JSON files, etc.).
- Extracting relevant data fields (neuron IDs, traces, time vectors, etc.).
- Cleaning and normalizing the data.
- Resampling the data to a common time resolution.
- Smoothing the data using different methods (e.g., Savitzky-Golay filter).
- Creating dictionaries to map neuron indices to neuron IDs and vice versa.
- Saving the preprocessed data in a standardized format.

## Customization

The submodule is designed to be easily customizable. You can modify the `_main.py` script to customize the data preprocessing process or add additional functionality. The `_utils.py` file contains utility functions and classes that can be modified as per your requirements.
