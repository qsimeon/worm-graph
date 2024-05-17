# Training Submodule

This submodule contains code for training and evaluating neural network model for neural data analysis.

## File Structure

The submodule consists of the following files:

- `_pkg.py`: Contains the main imports and configurations used in the training package.
- `_utils.py`: Contains utility functions for training and evaluating neural network model.
- `_main.py`: Provides main ML train loop learning and evaluation of the model.

This submodule uses the `data` and `model` submodules for loading the data and defining model.

## Usage

To use the training submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file `configs/submodule/train.yaml` to specify the training parameters.
3. Also specify the configurations of the desired datasets to use and model. Take a look into the `data` and `model` submodules for configuration.
4. Run the `python main.py +submodule=[dataset,model,train]` to start the training process using the specified configuration.
5. More usage examples are available in `configs` submodule.
6. The the model checkpoints and the learning curves will be stored into the `log/train` directory.

Note: Ensure that the dataset files are available in the specified locations and that the model configuration is compatible with the dataset.

## Training and Evaluation

The `_utils.py` module contains utility functions for training and evaluating neural network model. It includes functions for splitting data into train and test sets, optimizing model, and computing loss metrics. You can modify these functions to incorporate custom training or evaluation logic.

## Customization

The training submodule is designed to be easily customizable. You can modify the configuration file, data preprocessing modules, model configuration modules, and utility functions to adapt the training process to your specific needs.

Feel free to explore the code in each file for a more detailed understanding of the implementation and customization options.
