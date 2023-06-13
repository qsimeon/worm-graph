# Visualization Submodule

This submodule contains code for visualizing data from different sources, such as the calcium data, the connectome, and the neural network weights.

## File Structure

The submodule consists of the following files:

- `_pkg.py`: Contains the main imports and configurations used in the visualization package.
- `_utils.py`: Contains functions for visualizing data.
- `_main.py`: Contains the pipeline to run a visualization routine.

This submodule uses the `data` and `models` submodules for loading data and defining models to inspect.

## Usage

To use the visualization submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file `conf/visualize.yaml` to specify the visualization parameters.
3. Run the `_main.py` script to generate the plots.

Note: Ensure that the dataset files are available in the specified locations and that the model configuration is compatible with the dataset.

## Key Functions

1. `draw_connectome`: Plots the C. elegans hermaphrodite connectome network, provided as a PyTorch Geometric Data object, with color-coded neuron types and synaptic connections, and supports custom node positioning and labeling.
2. `plot_frequency_distribution`: Performs Fast Fourier Transform on input signal data and visualizes its frequency distribution on provided matplotlib axes.
3. `plot_loss_curves`: Generates and saves a plot of training and testing loss curves over epochs, utilizing data stored in a specified log directory.
4. `plot_before_after_weights`: Creates and saves a comparative visualization of the model's readout weights before and after the training process, utilizing information and model checkpoints from a specified log directory.
5. `plot_targets_predictions`: Generates and saves plots for the target and predicted neural activity for a specified (or all neurons) in a given (or all) worms, using prediction and target data from a specified log directory.
6. `plot_correlation_scatterplot`: Creates and saves scatterplots displaying the correlation between the target and predicted neural activity for a specified (or all) neurons in a given (or all) worms, where data points are colored based on whether they're from the training or testing set.
7. `plot_worm_data`:  Generates and displays a plot showing the calcium signal traces for a specified number of randomly selected neurons from the given worm's data, with the option to display either raw or smoothed traces.

Note: More details about the usage of these functions and the expected parameters can be found in the code comments.

## Customization

The visualization submodule is designed with flexibility in mind. You can modify the configuration file, data preprocessing modules, plotting functions, and utility functions to customize the visualization process according to your specific requirements.

We encourage you to delve into the code within each file for a deeper comprehension of the implementation and possible modifications.