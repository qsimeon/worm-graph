# worm-graph
## Simulating worms with graph nets.

```
├── data
   ├── batch_samler.py
   ├── graph_inject_data.py
   ├── load_connectome.py
   ├── load_neural_activity.py
   ├── map_dataset.py
   └── PlotRealData.ipynb
├── preprocess
   ├── export_nodes_edges.m
   ├── pickle_neural_data.py
   └── process_raw.py
├── models
   ├── gnn_models.py
   ├── linear_models.py
   └── rnn_models.py
├── tasks
   └── all_tasks.py
├── train
   ├── add_train_val_mask.py
   ├── GNNLossCurves.ipynb
   ├── LossBaselines.ipynb
   ├── 
   └── 
├── Visualizations
   ├── Connectome graph
      ├── DrawConnectome.ipynb
   ├── Loss curves
   └── Time series predictions
└── PipelineExplorer.ipynb: 
   └── Notebook to run through the pipeline interactively.
 ```
 
## Create the environment from the `environment.yml` file

`cd` into the `worm-graph` directory on your local machine: `cd worm-graph`

Using the terminal or an Anaconda Prompt: `conda env create -f environment.yml`
   The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Add the `worm-graph` root directory to Anaconda path: `conda develop .`
   *Important:* Do not skip the step above. Otherwise you will be faced with a lot of `ModuleNotFoundError`s.

Verify that the new environment was installed correctly: `conda env list`
   You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

## Get started with the pipeline in 1-line

`python -i main.py`

 ## Naming conventions
 
 For folders and script files, use the `lower_case_with_underscores` naming style.
 **Example:** `my_folder`, `my_script.py`.
 
 For Jupyter notebooks, use the `UPPER_CASE_WITH_UNDERSCORES` naming style.
 **Example:** `MyAnlysisNotebook.ipynb`.
 
 ## Style conventions

 * Aim to make every script not significantly longer than 100 lines. If your code is getting longer than this, it probably is a 
   good idea to modularize things by putting certain functions or classes in separare files like `utils.py` or `models.py`, etc.
 * Always shape neural data matrices as `(time, neurons, features)`.
 * Use the [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) code formatter. Before a commit, run the command `black .` in the Terminal to automatically reformat code according to the Black Code Style. 

 ## Organization: things to TODO.

- Do both: 
   - training on the first half of timesteps predicting the second half, and;
   - training on the second half of timesteps and the predicting the first half.
- Look at how people structure language models (NLP). They are tested on predicting arbitrary future timesteps. 
- Various tasks to implement:
   - predict the identity of the neuron given the trace (node prediction).
   - predict whether or not there exist an edge (edge prediction). 
   - predict the behavior of the worm from its neural activity.