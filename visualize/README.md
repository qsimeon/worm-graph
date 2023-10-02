# Visualization Submodule

This submodule contains code for visualizing data from different sources, such as calcium data, the connectome, predictions, and training metrics.

## File Structure

The submodule consists of the following files:

- `_pkg.py`: Contains the main imports and configurations used in the visualization package.
- `_utils.py`: Contains functions for visualizing data.
- `_main.py`: Contains the pipeline to run a visualization routine.

## Usage

To use the visualization submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file `configs/submodule/visualize.yaml` to specify the visualization parameters.
3. Run `python main.py +submodule=visualize` to generate the plots for a specified log directory, or
4. Run `python main.py +submodule=[dataset, model, ... , visualize]` to run the entire pipeline and then generate the plots automatically.
5. For more example usages, visit the `configs` submodule.

## Key Functionalities

This submodule is designed to be flexible, allowing for the plotting of individual runs or entire experiment runs automatically.

These are the plots for single runs:
- `plot_dataset_info()`: Plots neuron counts of training and validation datasets.
- `plot_loss_curves()`: Plots learning curves after training.
- `histogram_weights_animation()`: Plots an animation showing how the network weights change during training.
- `plot_predictions()`: Plots the predicted neural activity.
- `plot_pca_trajectory()`: Plots the PCA trajectory in 2D and 3D, comparing the predictions to the true signal.

For running an experiment (multiple runs), these are the available plots:
- `plot_exp_losses()`: Plots the validation loss of all runs in the experiment.
- `plot_scaling_law()`: Plots the scaling law related to the experiment (if applicable).

## Customization

The visualization submodule is designed with flexibility in mind. You can modify the configuration file, data preprocessing modules, plotting functions, and utility functions to tailor the visualization process to your specific requirements.

We encourage you to delve into the code within each file for a deeper understanding of the implementation and potential modifications.