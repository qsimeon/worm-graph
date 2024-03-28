# Preprocessing Submodule

This repository contains the functions used to preprocess the open-source calcium imaging data for neural activity analysis. The datasets are preprocessed and organized in a standard manner, making them ready to be used for various tasks such as neural network training, data analysis, and visualization.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the pipeline for preprocessing the data as specified in the configuration file.
- `_utils.py`: Contains utility functions and classes for data processing.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file in `configs/submodule/preprocess.yaml` as desired (see the explanation of each parameter in the configs submodule README).
3. Run the main pipeline: `python main.py +submodule=preprocess`.
4. The processed dataset will be saved in pickle format in the `data/processed/neural` folder.
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

- `source_dataset`: (str) Name of the dataset
- `smooth_method`: (str) Method used to smooth the calcium data
- `interpolate_method`: (std) Method used to interpolate the calcium data
- `worm`: (str) The worm ID in the COMBINED dataset (if you load more than one dataset)
- `original_worm`: (str) The worm ID in the original dataset (when you load a single dataset)
- `original_max_timesteps`: (int) Number of time steps before resampling
- `max_timesteps`: (int) Number of time steps after resampling
- `original_dt`: (torch.tensor) Column vector containing the difference between time steps (before resampling). Shape: (original_max_timesteps, 1)
- `dt`: (torch.tensor) Column vector containing the difference between time steps. Shape: (max_timesteps, 1)
- `residual_` and `original_calcium_data`: (torch.tensor) Standardized and normalized calcium data. Shape: (original_max_timesteps, 302)
- `residual_` and `calcium_data`: (torch.tensor) Standardized, normalized and resampled calcium data. Shape: (max_timesteps, 302)
- `residual_` and `original_smooth_calcium_data`: (torch.tensor) Standardized, smoothed and normalized calcium data. Shape: (original_max_timesteps, 302)
- `residual_` and `smooth_calcium_data`: (torch.tensor) Standardized, smoothed, normalized and resampled calcium data. Shape: (max_timesteps, 302)
- `original_time_in_seconds`: (torch.tensor) A column vector with the original time recording times (without resampling). Shape: (original_max_timesteps, 1)
- `time_in_seconds`: (torch.tensor) A column vector equally spaced by dt after resampling. Shape: (max_timesteps, 1)
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
- Resampling the data to a common time resolution. - if requested
- Smoothing the data using different methods (e.g., Savitzky-Golay filter). - if requested
- Creating dictionaries to map neuron indices to neuron IDs and vice versa.
- Saving the preprocessed data in a standardized format.

## Customization

The submodule is designed to be easily customizable. You can modify the `_main.py` script to customize the data preprocessing process or add additional functionality. The `_utils.py` file contains utility functions and classes that can be modified as per your requirements.
