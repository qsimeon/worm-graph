# worm-graph
## Simulating the _C. elegans_ whole brain with neural networks.

`tree -L 2 worm-graph/`
```
.
├── LICENSE
├── README.md
├── __init__.py
├── __pycache__
│   └── utils.cpython-39.pyc
├── analysis
│   ├── MPNsNeurogym.ipynb
│   ├── jpyEDM
│   └── takens
├── conf
│   ├── analysis.yaml
│   ├── dataset.yaml
│   ├── govfunc.yaml
│   ├── main.yaml
│   ├── model.yaml
│   ├── preprocess.yaml
│   ├── simulate.yaml
│   ├── slurm_default.yaml
│   ├── task.yaml
│   ├── test_config.yaml
│   ├── train.yaml
│   └── visualize.yaml
├── data
│   ├── _main.py
│   ├── _pkg.py
│   ├── _utils.py
│   ├── create_sine_dataset.py
│   ├── processed
│   └── raw
├── environment_complete.yml
├── environment_minimal.yml
├── govfunc
│   ├── Flavell2023
│   ├── Kaplan2020
│   ├── Kato2015
│   ├── Nichols2017
│   ├── Skora2018
│   ├── Uzel2022
│   ├── _main.py
│   ├── _pkg.py
│   ├── _utils.py
│   ├── correlation.png
│   ├── govfunc_lorenz
│   ├── rsa_analysis_Uzel2022.ipynb
│   ├── worm_response_pred.png
│   └── worm_response_target.png
├── main.py
├── models
│   ├── _main.py
│   ├── _pkg.py
│   └── _utils.py
├── pkg.py
├── preprocess
│   ├── __pycache__
│   ├── _main.py
│   ├── _pkg.py
│   ├── _utils.py
│   └── export_nodes_edges.m
├── tasks
│   ├── _main.py
│   ├── _pkg.py
│   └── _utils.py
├── tempCodeRunnerFile.py
├── testing
│   ├── FuturePredictionCElegansNNs.ipynb
│   ├── GNNLossCurves.ipynb
│   ├── LossBaselines.ipynb
│   ├── PlotRealData.ipynb
│   ├── analyze_logs_test.ipynb
│   ├── ivy_scripts
│   ├── quick_script_1.py
│   ├── quick_script_2.py
│   ├── quick_script_3.py
│   ├── quick_script_5.py
│   ├── quick_script_6.py
│   ├── tempCodeRunnerFile.py
│   └── test_config.py
├── train
│   ├── _main.py
│   ├── _pkg.py
│   ├── _utils.py
│   └── train_vis_main.py
├── utils.py
└── visualization
    ├── DrawConnectome.ipynb
    ├── PipelineExplorer.ipynb
    ├── _main.py
    ├── _pkg.py
    ├── _utils.py
    ├── plot_for_atlas
    └── tempCodeRunnerFile.py
 ```
 
## Create the environment from the `environment.yml` file

`cd` into the `worm-graph` directory on your local machine: `cd worm-graph`

Using the terminal or an Anaconda Prompt: `conda env create -f environment_minimal.yml`
   <br>The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Add the `worm-graph` root directory to Anaconda path: `conda develop .`
   <br>*Important:* Do not skip this step step! Otherwise you will be faced with several `ModuleNotFoundError`.

Verify that the new environment was installed correctly: `conda env list`
   <br>You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

## Get started with the pipeline in one line

`python -u main.py`

For one multi-worm dataset of neural activity, this pipeline will:
1. Load the preprocessed calcium data for all worms in the dataset.
2. Train a neural network model to predict future calcium activity from previous activity.
3. Plot the train and validation loss curves for the model and its predictions on test data.

## TODO: How to add models and datasets


## File naming conventions

For folders and script files, use the `lower_case_with_underscores` naming style.
**Example:** `my_folder`, `my_script.py`.

For Jupyter notebooks, use the `CamelCase` naming style.
**Example:** `MyAnalysisNotebook.ipynb`.

## Code style conventions

Aim to make every runnable script (e.g. Python files with a `if __name__ == "__main__":` section) not significantly longer than 100 lines. If your code is getting longer than this, it probably is a good idea to modularize things by encapsulating certain processes in helper functions and moving those to a separare file like `_utils.py`. 

Note the orgaization structure of this project. Each self-contained (sub-)module is in its own folder containing the files: `_main.py`, `_utils.py` and `_pkg.py`. 
   - `_main.py` holds the main code that module executes, typically as a single function that gets called in `if __name__ == "__main__":` part. 
   - `_pkg.py` is exclusively for placing all package imports that the module needs. 
   - `_utils.py` is the bulk of the module's code as it contains the definitions for all custom classes and helper functions to be used by the module.
   - there may be other miscellaneous subfolders and files that are important and specific to that module (e.g. the `processed/` and `raw/` folders inside `data/`) or just something that is a work in progress.

Use the [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) formatter. Before a commit, run the command `black .` in the Terminal from the repository's root directory `worm-graph`. This will automatically reformat all code according to the [Black Code Style](https://github.com/psf/black). 

Use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for comments, documentation strings and unit tests.

When in doubt about anything else style related that's not addressed by the previous two points, reference the [Python Enhancement Protocols (PEP8)](https://peps.python.org/pep-0008/).

Always shape neural data matrices as `(time, neurons, [features])`. The braces `[]` indicate that the last `features` dimension is optional, as the `neurons` currently serve as the features for our models. 

## Things to TODO.

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
