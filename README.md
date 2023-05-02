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

`cd` into the `worm-graph` directory on your local machine: `cd worm-graph`

Using the terminal or an Anaconda Prompt: `conda env create -f environment_cluster.yml`
   <br>The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Add the `worm-graph` root directory to Anaconda path: `conda develop .`
   <br>*Important:* Do not skip this step step! Otherwise you will be faced with several `ModuleNotFoundError`.

Verify that the new environment was installed correctly: `conda env list`
   <br>You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

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

<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

## Future Tasks

- Urgent TODOs: 
   - scaling law plots.
   - search-based approach to logging .
- Less urgent TODOs: 
   - investigate the how large language models for NLP are structured to allow prediction of arbitrary future timesteps.
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
      - training on a single worm vs. multiple worms.
