# worm-graph
## Simulating worms with graph nets.

```
├── analysis/
   ├── jpyEDM/
   ├── takens/
   ├── MPNsNeurogym.ipynb
├── conf/
   ├── analysis.yaml
   ├── dataset.yaml
   ├── main.yaml
   ├── model.yaml.py
   ├── preprocess.yaml
   ├── simulate.yaml
   ├── task.yaml
   ├── train.yaml
   └── visualize.yaml
├── data/
   ├── processes/
   ├── raw/
   ├── _main.py
   ├── _pkg.py
   ├── _utils.py
   └── PlotRealData.ipynb
├── logs/
   ├── multirun/
   └── run/
├── models
   ├── _main.py
   ├── _pkg.py
   └── _utils.py
├── preprocess
   ├── _main.py
   ├── _pkg.py
   ├── _utils.py
   └── export_nodes_edges.m
├── tasks
   ├── _main.py
   ├── _pkg.py
   └── _utils.py
├── train
   ├── _main.py
   ├── _pkg.py
   ├── _utils.py
   ├── GNNLossCurves.ipynb
   └── LossBaselines.ipynb
├── visualization
   ├── _main.py
   ├── _pkg.py
   ├── _utils.py
   ├── DrawConnectome.ipynb
   └── PipeLineExplorer.ipynb
├── __init__.py
├── main.py
├── pkg.py
├── main.py
├── utils.py
├── quick_script_1.py
├── quick_script_2.py
├── quick_script_3.py
├── quick_script_4.py
├── README.md
└── FuturePredictionCElegansNNs.ipynb
 ```
 
## Create the environment from the `environment.yml` file

`cd` into the `worm-graph` directory on your local machine: `cd worm-graph`

Using the terminal or an Anaconda Prompt: `conda env create -f environment.yml`
   <br>The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Add the `worm-graph` root directory to Anaconda path: `conda develop .`
   <br>*Important:* Do not skip this step step! Otherwise you will be faced with several `ModuleNotFoundError`.

Verify that the new environment was installed correctly: `conda env list`
   <br>You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

## Get started with the pipeline in 1-line

`python -i main.py`

For each of several multi-worm calcium imaging datasets, this pipeline will:
1. Load the preprocessed calcium data for all worms in the dataset.
2. Train a neural network model to predict future calcium activity from previous activity.
3. Plot the train and validation loss curves for the model and its predictions on test data.

## Naming conventions

For folders and script files, use the `lower_case_with_underscores` naming style.
**Example:** `my_folder`, `my_script.py`.

For Jupyter notebooks, use the `UPPER_CASE_WITH_UNDERSCORES` naming style.
**Example:** `MyAnalysisNotebook.ipynb`.

## Style conventions

Aim to make every runnable script (e.g. Python files with a `if __name__ == "__main__":` section) not significantly longer than 100 lines. If your code is getting longer than this, it probably is a good idea to modularize things by encapsulating certain processes in helper functions and moving those to a separare file like `_utils.py`. 

Note the orgaization structure of this project. Each self-contained (sub-)module is in its own folder with `_main.py`, `_utils.py` and `_pkg.py`. `_main.py` holds the main code that module executes, typically as a single function that gets called in `if __name__ == "__main__":` part. `_pkg.py` is exclusively for placing all package imports that the module needs. `_utils.py` is the bulk of the module's code as it contains the definitions for all custom classes and helper functions to be used by the module.

Use the [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) formatter. Before a commit, run the command `black .` in the Terminal from the repository's root directory `worm-graph`. This will automatically reformat all code according to the [Black Code Style](https://github.com/psf/black). 

Use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for comments, documentation strings and unit testing
When in doubt about anything else style related that's not addressed by the previous two points, reference the [Python Enhancement Protocols (PEP8)](https://peps.python.org/pep-0008/).

Always shape neural data matrices as `(time, neurons, [features])`. The braces `[]` indicate that the last `features` dimension is optional, as the `neurons` currently serve as the features for our models. 

## Organization: things to TODO.

- Urgent TODOs: 
   - scaling law plots.
   - search-based approach to logging .
- Less urgent TODOs: 
   - investigate the how large language models for NLP are structure to allow prediction of arbitrary future timesteps.
- Think about canonical plots to always make:
   - hold all else constant except for a single parameter / config item.
   - color lines by values of the varied parameter / config item.
- Various tasks to implement:
   - predict the identity of the neuron given the trace (node prediction).
   - predict whether or not there exist an edge (edge prediction). 
   - predict the behavior of the worm from its neural activity.
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

