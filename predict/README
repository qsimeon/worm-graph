# Predict Submodule

This submodule contains the code for making predictions on a dataset with a trained model and saving the outputs into a .csv file.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the main script for making predictions and saving them into a .csv file.
- `_utils.py`: Contains utility functions for making predictions using the trained model.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the prediction configuration file `conf/predict.yaml` to specify the model and other parameters.
3. Run the `_main.py` script to make predictions with the trained model and save them into a .csv file. This script calls the necessary functions from `_utils.py`.
4. Use the prediction results for further analysis or processing in your application.

Note: Make sure to have the trained model file in the appropriate directory before running the code. 

### Function Structure

The `_utils.py` file provides a function `model_predict()` to make predictions for all neurons on a dataset with a trained model.
The `model_predict()` function takes a trained model and a tensor of calcium data, and returns a tuple of three tensors: inputs, predictions, and targets. Each tensor has the same shape as the input calcium data.

## Customization

The submodule is designed to be easily customizable. You can modify the `_main.py` script to customize the prediction process or add additional functionality. The `_utils.py` file contains utility functions that can be modified as per your requirements.
