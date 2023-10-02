# Predict Submodule

This submodule contains the code for making predictions on a dataset using a trained model and saving the outputs to a .csv file.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the main script for making predictions and saving them to a .csv file.
- `_utils.py`: Contains utility functions for making predictions with the trained model.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the prediction configuration file `configs/submodule/predict.yaml` to specify the parameters (see the `configs` submodule for more information).
3. Also, configure `configs/submodule/dataset.yaml` and `configs/submodule/model.yaml` to select which model and data to use.
4. Run `python main.py +submodule=[dataset, model, predict]` to make predictions with the trained model and save them to a .csv file in `logs/predict`.
5. Use the prediction results for further analysis or processing in your application.
6. For more usage examples, refer to the `configs` submodule.

## Customization

The submodule is designed to be easily customizable. You can modify the `_main.py` script to tailor the prediction process or add additional functionality. The `_utils.py` file contains utility functions that can be adjusted according to your requirements.