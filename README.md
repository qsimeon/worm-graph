# worm-graph

Simulating the _C. elegans_ whole brain with neural networks.

---

### Notes to Users
- Please change `submitit_slurm` to `submitit_local` in the file `configs/pipeline.yaml` if running on your local machine.

- Disable autocast if using a CPU.

### Cite
Q. Simeon, L. Venâncio, M. A. Skuhersky, A. Nayebi, E. S. Boyden and G. R. Yang, "Scaling Properties for Artificial Neural Network Models of a Small Nervous System," SoutheastCon 2024, Atlanta, GA, USA, 2024, pp. 516-524, doi: 10.1109/SoutheastCon52093.2024.10500049.

Simeon, Q., Kashyap, A., Kording, K. P., & Boyden, E. S. (2024). Homogenized _C. elegans_ Neural Activity and Connectivity Data. In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/2411.12091

### [Hugging Face Dataset](https://huggingface.co/datasets/qsimeon/celegans_neural_data)

## Table of Contents
1. [Project Overview](#worm-graph)
2. [Environment Setup](#environment-setup)
    - [Recommended Bash Script](#recommended-bash-script)
3. [Getting Started](#getting-started)
4. [Folder Structure](#folder-structure)
    - [Directory Tree](#directory-tree)
    - [Submodule Information](#submodule-information)
5. [For Developers](#for-developers)
    - [File Naming Conventions](#file-naming-conventions)
    - [Code Style Conventions](#code-style-conventions)
6. [Future Tasks](#future-tasks--todos)

## Project Overview

`worm-graph` is a computational framework for modeling the neural dynamics of _Caenorhabditis elegans_ (_C. elegans_) using artificial neural networks (ANNs). 

## Environment Setup

To prepare your environment for this project, you should use the provided `setup/env.sh` bash script. 

### Recommended Bash Script 

1. Run the `setup/env.sh` script. This will create the new `worm-graph` environment and install the required packages:

    ```
    bash setup/env.sh
    ```

2. Activate the new `worm-graph` environment:

    ```
    conda activate worm-graph
    ```

3. After finishing the installations above, from the root directory (`worm-graph/`),  run:

    ```
    conda develop .
    ```

**Note:** 
* Please ensure to carry out that last step; otherwise you may encounter `ModuleNotFoundError` later on.

* You can check if the environment was successfully installed by running `conda env list` or `conda info --envs`. 
    
* Always activate the environment before starting your work on the project by running `conda activate worm-graph`.

* To use the OpenAI text embeddings, add your API key to the `.env` file. The file `worm-graph/.env` should look like this:
    ```
    # Once you add your API key below, make sure to not share it with anyone! The API key should remain private.
    OPENAI_API_KEY=abc123 # replace with your personal API key
    ```

## Getting Started

To make sure nothing breaks, the first thing you need to do is download and preprocess our curated collection of _C. elegans_ neural activity datasets. From the root (`worm-graph`) directory, run the command:

`python main.py +submodule=[preprocess]`

* If chaining multiple submodules together (e.g. `data` and `preprocess`) do not use spaces after commas:
    
    - `python main.py +submodule=[preprocess,data]`

* If on a Mac, place `+submodule=[preprocess]` in quotations:
    
    - `python main.py "+submodule=[preprocess]"`

    - `python main.py "+submodule=[preprocess,data]"`


Now you can run the main script as a demo of the full functional pipeline (`preprocess`) -> `data` -> `model` -> `train` -> `analysis` -> (`visualize`):

`python main.py +experiment=default_run`

* If on a Mac, place `+experiment=default_run` in quotes:

    - `python main.py "+experiment=default_run"`

* If running on Linux or a SLURM computing cluster:

    - `python main.py +experiment=default_multirun`

For one multi-worm dataset of neural activity, this pipeline will:
1. Load the preprocessed calcium data for all worms in the dataset.
2. Train a neural network model to predict future calcium activity from previous activity.
3. Plot the train and validation loss curves for the model, and its predictions on validation data.

## Folder Structure

### Directory Tree

`cd worm-graph`
`tree -L 1 .`
```
├── LICENSE
├── README.md
├── __init__.py
├── __pycache__
├── analysis
├── configs
├── data
├── logs
├── main.py
├── model
├── pkg.py
├── predict
├── preprocess
├── pyproject.toml
├── setup
├── train
├── utils.py
└── visualize
```

### Submodule Information
- `configs`
    - All experiment, evaluation, etc. config files compatible with hydra for streamlined development/experimental process
- `analysis`
    - Different analysis notebooks to identify valuable statistics to validate predictive capabilities of model
- `data`
    - Dataset class implementations, notebooks for validation + synthetic data generation
- `logs`
    - Hydra logs for runs
- `model`
    - Contains all model component implementation + loading util functions, package imports, wrapper for retrieving model architecture
- `opensource_data`
    - Datasets compiled from different experimental open source publications.
- `predict`
    - Prediction utilities - loading model, package imports, prediction passthrough
- `preprocess`
    - Preprocess data utilities
- `train`
    - Utilities for early stopping, saving checkpoints. Example training script for regressing calcium neural activity + validation during training
- `visualize`
    - Visualization notebooks - visualizing connectome + plotting neural activity

## For Developers

### File Naming Conventions

For folders and script files, use the `lower_case_with_underscores` naming style.
**Example:** `my_folder/my_script.py`.

For Jupyter notebooks, use the `CamelCase` naming style.
**Example:** `MyAnalysisNotebook.ipynb`.

### Code Style Conventions

1. Aim to keep every runnable script (e.g. Python files with a `if __name__ == "__main__":` section) not significantly longer than 300 lines. If your code is getting longer than this, consider modularizing by encapsulating certain processes in helper functions and moving them to a separate file like `_utils.py`.

2. Follow the organization structure of this project, where each self-contained (sub-)module is its own directory containing the files `_main.py`, `_utils.py`, and `_pkg.py`.
  - `_main.py` holds the main code that the module executes, typically as a single function that gets called in the `if __name__ == "__main__":` part.
  - `_pkg.py` is exclusively for placing all package imports that the module needs.
  - `_utils.py` contains the definitions for all custom classes and functions to be used by the module.

3. Use the [Black Code Style](https://github.com/psf/black) formatter:
  - Before committing, run the command `black .` in the Terminal from the repository's root directory `worm-graph`.
  - This will automatically reformat all code according to the Black Code Style.

4. Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for comments, documentation strings, and unit tests.

5. When in doubt about anything else style-related that's not addressed by the previous two points, reference the [Python Enhancement Protocols (PEP8)](https://peps.python.org/pep-0008/).

6. Always shape neural data matrices as `(time, neurons, {features})`. The braces `{}` indicate that the last `features` dimension is optional, as the `neurons` currently serve as the features for our model.

## Future Tasks

- Post the preprocess datasets to a file-hosting site.
- Implement unit tests for all submodules in the `tests` directory.
- Add docstrings to all functions and classes, follow the Google Python Style Guide for formating.
