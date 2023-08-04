# worm-graph
## Simulating the _C. elegans_ whole brain with neural networks.

`tree -L 1 worm-graph`
```
├── analysis
├── configs
├── data
├── debugging
├── __init__.py
├── LICENSE
├── main.py
├── models
├── opensource_data
├── pkg.py
├── predict
├── preprocess
├── __pycache__
├── pyproject.toml
├── README.md
├── setup
├── tests
├── train
├── utils.py
└── visualize
```
## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Getting Started](#getting-started)
3. [File Naming Conventions](#file-naming-conventions)
4. [Code Style Conventions](#code-style-conventions)
5. [Future Tasks](#future-tasks)

## Environment Setup

To prepare your environment for this project, you can use either the provided `.yml` files or the `init.sh` script. Choose the option that works most efficiently for your system.

Firstly, navigate to the `setup` directory. This directory contains all the necessary configuration files to set up the virtual environment. Use the following command to access the directory:

```
cd setup
```

**Note:** Installing the environment can take up to 2 hours!

## Option A: Setting up with the `.yml` files

1. Open your terminal or Anaconda Prompt, and create a new Conda environment using the `.yml` file. The first line of the `.yml` file sets the name of the new environment.

    ```
    conda env create -f environment.yml
    ```

## Option B: Setting up with the bash script (recommended)

1. Run the `env.sh` script. This will create the new `worm-graph` environment and install the required packages:

    ```
    bash env.sh
    ```

2. Activate the new `worm-graph` environment:

    ```
    conda activate worm-graph
    ```

3. Install the remaining dependencies using pip:

    ```
    pip install --upgrade -r requirements.txt
    ```

After finishing one of the installations above, navigate back to the root directory (`worm-graph/`) and run:

    conda develop .

**Note:** Please ensure to carry out this step, otherwise you may encounter `ModuleNotFoundError`.

*You can check if the environment was successfully installed by running `conda env list` or `conda info --envs`.*

**Important Reminder:** Always activate the environment before starting your work on the project using `conda activate worm-graph`.

## Getting Started

To make sure nothing breaks, the first thing you need to do is download and preprocess our curated collection of _C. elegans_ neural activity datasets. 
From the root (`worm-graph`) directory, run the command:

`python main.py +submodule=[preprocess]`

Now you can run the main script as a demo of the fully functional pipeline:

`python main.py +experiment=default_run`

If you are running on a SLURM computing cluster:

`python main.py +experiment=default hydra/launcher=submitit_slurm`

For one multi-worm dataset of neural activity, this pipeline will:
1. Load the preprocessed calcium data for all worms in the dataset.
2. Train a neural network model to predict future calcium activity from previous activity.
3. Plot the train and validation loss curves for the model and its predictions on test data.

## File Naming Conventions

For folders and script files, use the `lower_case_with_underscores` naming style.
**Example:** `my_folder`, `my_script.py`.

For Jupyter notebooks, use the `CamelCase` naming style.
**Example:** `MyAnalysisNotebook.ipynb`.

## Code Style Conventions

- Aim to keep every runnable script (e.g. Python files with a `if __name__ == "__main__":` section) not significantly longer than 300 lines. If your code is getting longer than this, consider modularizing by encapsulating certain processes in helper functions and moving them to a separate file like `_utils.py`.

- Follow the organization structure of this project, where each self-contained (sub-)module has its own folder containing the files: `_main.py`, `_utils.py`, and `_pkg.py`.
  - `_main.py` holds the main code that the module executes, typically as a single function that gets called in the `if __name__ == "__main__":` part.
  - `_pkg.py` is exclusively for placing all package imports that the module needs.
  - `_utils.py` contains the definitions for all custom classes and helper functions to be used by the module.

- Use the [Black Code Style](https://github.com/psf/black) formatter:
  - Before committing, run the command `black .` in the Terminal from the repository's root directory `worm-graph`.
  - This will automatically reformat all code according to the Black Code Style.

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for comments, documentation strings, and unit tests.

- When in doubt about anything else style-related that's not addressed by the previous two points, reference the [Python Enhancement Protocols (PEP8)](https://peps.python.org/pep-0008/).

- Always shape neural data matrices as `(time, neurons, [features])`. The braces `[]` indicate that the last `features` dimension is optional, as the `neurons` currently serve as the features for our models.

## Future Tasks / TODOs:

- Perform scaling experiments:
      - Vary the training dataset size.
      - Create scaling law plots.
- Add unit tests for each submodule.
- Get networks to perform better than the baseline.
- Add docstrings to all functions and classes in all submodules.
   - Format using the Google Python Style Guide.
