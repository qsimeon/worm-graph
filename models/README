# Models Submodule

This submodule contains code for defining and utilizing various models for neural data analysis.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the main script for loading the model specified in the configuration file, with its respective hyperparameters.
- `_utils.py`: Contains the implementation of the models.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file `conf/model.yaml` to specify the model type, size, and other relevant parameters.
3. Run the `_main.py` script to instantiate or load the model based on the provided configuration.

Note: Ensure that the required checkpoints or model files are available in the specified locations if loading a saved model.

## Model Classes

The `_utils.py` script includes several model classes, which are subclasses of the `Model` superclass. Each model class represents a different type of neural network model and provides specific implementation details.

The available model classes are:

- `LinearNN`: A simple linear regression model used as a baseline.
- `NeuralTransformer`: A neural transformer model.
- `NeuralCFC`: A neural circuit policy (NCP) closed-form continuous time (CfC) model.
- `NetworkLSTM`: A model of the _C. elegans_ neural network using an LSTM.

You can select the desired model type in the configuration file `conf/model.yaml` by specifying the `type` parameter.

## Customization

The submodule is designed to be easily customizable. You can modify the `get_model` function in `_main.py` to customize the model pipeline or add additional functionality. The `_utils.py` file contains utility functions and classes that can be modified according to your requirements.

Feel free to explore the code in each file for a more detailed understanding of the implementation and customization options.
