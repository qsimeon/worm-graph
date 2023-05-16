# worm-graph
## Simulating the _C. elegans_ whole brain with neural networks.

`tree -L 2 worm-graph/`
```
├── LICENSE
├── README.md
├── __init__.py
├── __pycache__
├── analysis
├── conf
├── data
├── environment_cluster.yml
├── environment_local.yml
├── govfunc
├── logs
├── main.py
├── models
├── pkg.py
├── preprocess
├── testing
├── train
├── utils.py
└── visualization
```
## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Getting Started](#getting-started)
3. [File Naming Conventions](#file-naming-conventions)
4. [Code Style Conventions](#code-style-conventions)
5. [Future Tasks](#future-tasks)

## Environment Setup

To prepare your environment for this project, you can use either the provided `.yml` files or the `init.sh` script. Choose the option that works most efficiently for your system.

Firstly, navigate to the `env_setup` directory. This directory contains all the necessary configuration files to set up the virtual environment. Use the following command to access the directory:

```
cd worm-graph/env_setup
```

## Option A: Setting up with the `.yml` files

1. Open your terminal or Anaconda Prompt, and create a new Conda environment using the `.yml` file. The first line of the `.yml` file sets the name of the new environment.

    ```
    conda env create -f environment.yml
    ```


## Option B: Setting up with the bash script (recommended)

**Note:** Installing the environment can take anywhere between 10 to 30 minutes.

1. Run the `env.sh` script. This will create the new `worm-graph` environment and install the required packages:

    ```
    source env.sh
    ```

2. Activate the new `worm-graph` environment:

    ```
    conda activate worm-graph
    ```

3. Install the remaining dependencies using pip:

    ```
    pip install -r requirements.txt
    ```

After finishig one of the installations above, navigate back to the root directory (`worm-graph/`) and run:

    ```
    conda develop .
    ```

**Note:** Please ensure to carry out this step, otherwise you may encounter `ModuleNotFoundError`.

*You can check if the environment was successfully installed by running `conda env list` or `conda info --envs`.*

**Important Reminder:** Always activate the environment before starting your work on the project using `conda activate worm-graph`.

## Getting Started

`python -u main.py`

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

- Aim to keep every runnable script (e.g. Python files with a `if __name__ == "__main__":` section) not significantly longer than 100 lines. If your code is getting longer than this, consider modularizing by encapsulating certain processes in helper functions and moving them to a separate file like `_utils.py`.

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

<!-- [![[Black Code Style](https://github.com/psf/black)](https://img.shields.io/badge/code%20style-black-000000.svg)] -->

## Future Tasks

- Urgent TODOs: 
   - scaling law plots.
   - search-based logging.
- Think about canonical plots to always make:
   - hold all else constant except for a single parameter / config item.
   - color lines by values of the varied parameter / config item.
- Various tasks to implement:
   - predict the identity of a neuron given neural activity (node prediction).
   - predict whether or not a pair of neurons are connected (edge prediction). 
   - predict the instantaneous behavior of a worm given its current and recent neural activity.
- Goals for the future:
   - get networks to perform better than the baseline.
   - better documentation:
      - add a description of project-specific terminology to this README
      - add unit tests for each submodule.
      - add docstrings to all functions using Google Python Style Guide.
   - implement graph neural network (GNN models):
      - with connectome constraint.
      - without connectome constraint.
      - is additional biological data needed?
   - perform scaling experiments:
      - varying the (train) dataset size.
      ~~- training on a single worm vs. multiple worms.~~
