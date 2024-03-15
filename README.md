# worm-graph

### Simulating the _C. elegans_ whole brain with neural networks.

# Cite
Simeon, Quilee, Leandro Venâncio, Michael A. Skuhersky, Aran Nayebi, Edward S. Boyden, and Guangyu Robert Yang. 2024. “Scaling Properties for Artificial Neural Network Models of a Small Nervous System.” bioRxiv. https://doi.org/10.1101/2024.02.13.580186.

@UNPUBLISHED{Simeon2024-dy,
  title    = "Scaling Properties for Artificial Neural Network Models of a
              Small Nervous System",
  author   = "Simeon, Quilee and Ven{\^a}ncio, Leandro and Skuhersky, Michael A
              and Nayebi, Aran and Boyden, Edward S and Yang, Guangyu Robert",
  abstract = "The nematode worm C. elegans provides a unique opportunity for
              exploring in silico data-driven models of a whole nervous system,
              given its transparency and well-characterized nervous system
              facilitating a wealth of measurement data from wet-lab
              experiments. This study explores the scaling properties that may
              govern learning the underlying neural dynamics of this small
              nervous system by using artificial neural network (ANN) models.
              We investigate the accuracy of self-supervised next time-step
              neural activity prediction as a function of data and models. For
              data scaling, we report a monotonic log-linear reduction in
              mean-squared error (MSE) as a function of the amount of neural
              activity data. For model scaling, we find MSE to be a nonlinear
              function of the size of the ANN models. Furthermore, we observe
              that the dataset and model size scaling properties are influenced
              by the particular choice of model architecture but not by the
              precise experimental source of the C. elegans neural data. Our
              results fall short of producing long-horizon predictive and
              generative models of C. elegans whole nervous system dynamics but
              suggest directions to achieve those. In particular our data
              scaling properties extrapolate that recording more neural
              activity data is a fruitful near-term approach to obtaining
              better predictive ANN models of a small nervous system. \#\#\#
              Competing Interest Statement The authors have declared no
              competing interest.",
  journal  = "bioRxiv",
  pages    = "2024.02.13.580186",
  month    =  mar,
  year     =  2024,
  language = "en"
}


# NOTE to Users! Use code from this commit: 3495d274d0d4c06209899fd68a62760bdd09101c 
We are currently working on changes involving adding a ninth dataset in the main branch so the current commit will not run!
We apologize for the incovenience and expect to have working code after publication by the end of March 2024!

### Table of Contents
1. [Project Overview](#worm-graph)
2. [Directory Structure](#directory-structure)
3. [Environment Setup](#environment-setup)
4. [Getting Started](#getting-started)
5. [For Developers](#for-developers)
    - [File Naming Conventions](#file-naming-conventions)
    - [Code Style Conventions](#code-style-conventions)
6. [Future Tasks](#future-tasks--todos)


## Project Overview

`worm-graph` is a computational framework for modeling and simulating the neural dynamics of _Caenorhabditis elegans_ (_C. elegans_) using artificial neural networks (ANNs). The project focuses on self-supervised learning to predict future neural activity from historical data without behavioral context. It employs a range of neural network architectures such as LSTM, Transformer, and Feed-Forward networks, evaluating their performance based on mean squared error (MSE) as they scale with training data volume and model complexity.

This repository serves as a platform to investigate the effects of training data size, network architecture, and model parameters on the accuracy of neural state predictions. It utilizes diverse datasets, recorded under various conditions, to ensure robustness and generalizability of the models. A key finding of this work is the logarithmic reduction in MSE with increased training data, highlighting the critical role of data volume in model performance. The research also reveals a nonlinear dependency of prediction accuracy on model size, identifying an optimal range for the number of trainable parameters.


## Directory Structure
`tree -L 1 worm-graph`
```
├── analysis
├── configs
├── data
├── __init__.py
├── LICENSE
├── logs
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


## Environment Setup

To prepare your environment for this project, you should use either the provided `env.sh` script. Firstly, navigate to the `setup` directory. This directory contains all the necessary configuration files to set up the virtual environment. Use the following command to access the directory:

```
cd setup
```

**Note:** Installing the environment can sometimes take up to 1 hour!


### Setting up with the bash script (recommended) 

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


4. After finishing the installations above, navigate back to the root directory (`worm-graph/`):
    ```
    cd ..
    ```

    then run:

    ```
    conda develop .
    ```

**Note:** 
* Please ensure to carry out this step, otherwise you may encounter `ModuleNotFoundError` later on.

* You can check if the environment was successfully installed by running `conda env list` or `conda info --envs`. 

**Important Reminder:** Always activate the environment before starting your work on the project by running `conda activate worm-graph`.


## Getting Started

To make sure nothing breaks, the first thing you need to do is download and preprocess our curated collection of _C. elegans_ neural activity datasets. From the root (`worm-graph`) directory, run the command:

`python main.py +submodule=[preprocess]`

* If chaining multiple submodules together (e.g. `dataset` and `preprocess`) do not use spaces after commas:
    
    - `python main.py +submodule=[preprocess,dataset]`

* If on a Mac, place `+submodule=[preprocess]` in quotations:
    
    - `python main.py "+submodule=[preprocess]"`

    - `python main.py "+submodule=[preprocess,dataset]"`


Now you can run the main script as a demo of the fully functional pipeline:

`python main.py +experiment=default_run`

* If on a Mac, place `+experiment=default_run` in quotations:

    - `python main.py "+experiment=default_run"`

* If you are running on a SLURM computing cluster:

    - `python main.py +experiment=default_multirun`


For one multi-worm dataset of neural activity, this pipeline will:
1. Load the preprocessed calcium data for all worms in the dataset.
2. Train a neural network model to predict future calcium activity from previous activity.
3. Plot the train and validation loss curves for the model, and its predictions on validation data.

For more tutorials on how to use the pipeline and configuration files, refer to the `worm-graph` GitHub Wiki page.


## For Developers


### File Naming Conventions

For folders and script files, use the `lower_case_with_underscores` naming style.
**Example:** `my_folder/my_script.py`.

For Jupyter notebooks, use the `CamelCase` naming style.
**Example:** `MyAnalysisNotebook.ipynb`.


### Code Style Conventions

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

- Always shape neural data matrices as `(time, neurons, {features})`. The braces `{}` indicate that the last `features` dimension is optional, as the `neurons` currently serve as the features for our models.


## Future Tasks

- Post the preprocess datasets to a file-hosting site.
- Implement unit tests for all submodules in the `tests` directory.
- Add docstrings to all functions and classes. Follow the Google Python Style Guide for formating.
