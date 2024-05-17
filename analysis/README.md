# Analysis Submodule

This submodule contains code for performing data analysis, such as clustering analysis, feature extraction, and statistical tests on various types of data, including calcium data, connectome, and neural network weights.

## File Structure

The submodule consists of the following files:

- `_pkg.py`: Contains the main imports and configurations used in the analysis package.
- `_utils.py`: Contains functions for carrying out different types of data analysis.
- `_main.py`: Contains the pipeline to execute an analysis routine.

This submodule uses the `data` and `model` submodules for loading the data and defining model to analyze.

## Usage

To use the analysis submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file `conf/analysis.yaml` to specify the analysis parameters.
3. Run the `_main.py` script to start the analysis routine.

Note: Ensure that the dataset files are available in the specified locations and that the model configuration is compatible with the dataset.

## Key Functions

*Under development.*

## Customization

The analysis submodule is designed with flexibility in mind. You can modify the configuration file, data preprocessing modules, analysis functions, and utility functions to adapt the analysis process according to your specific needs.

Feel free to explore the code in each file for a more detailed understanding of the implementation and customization options.