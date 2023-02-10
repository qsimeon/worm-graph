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
   ├── lstm_hidden_size_experiment.py
   ├── scaling_train.py
   ├── train_gnn.py
   └── train_main.py
├── Visualizations
   ├── draw_connectome.py
   ├── DrawConnectome.ipynb
   ├── PipeLineExplorer.ipynb
   ├── plot_before_after_weights.py
   ├── plot_correlation_scatter.py
   ├── plot_loss_log.py
   ├── plot_more_data_losses.py
   ├── plot_multi_worm_losses.py
   ├── plot_neuron_train_test_samples.py
   ├── plot_single_neuron_signals.py
   ├── plot_target_prediction.py
   └── plot_worm_data.py
 ```
 
## Create the environment from the `environment.yml` file

`cd` into the `worm-graph` directory on your local machine: `cd worm-graph`

Using the terminal or an Anaconda Prompt: `conda env create -f environment.yml`
   <br>The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Add the `worm-graph` root directory to Anaconda path: `conda develop .`
   <br>*Important:* Do not skip this step step! Otherwise you will be faced with several `ModuleNotFoundError`s.

Verify that the new environment was installed correctly: `conda env list`
   <br>You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

## Get started with the pipeline in 1-line

`python -i main.py`
   <br> For each of several multi-worm calcium imaging datasets, this pipeline will:
      1. Load the preprocessed calcium data for all worms in the dataset.
      2. Train a neural network model to predict future calcium activity from previous activity.
      3. Plot the train and validation loss curves for the model and its predictions on test data.

 ## Naming conventions
 
 For folders and script files, use the `lower_case_with_underscores` naming style.
 **Example:** `my_folder`, `my_script.py`.
 
 For Jupyter notebooks, use the `UPPER_CASE_WITH_UNDERSCORES` naming style.
 **Example:** `MyAnalysisNotebook.ipynb`.
 
 ## Style conventions

 * Aim to make every script not significantly longer than 100 lines. If your code is getting longer than this, it probably is a 
   good idea to modularize things by putting certain functions or classes in separare files like `utils.py` or `models.py`, etc.
 * Always shape neural data matrices as `(time, neurons, features)`.
 * Use the [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) code formatter. Before a commit, run the command `black .` in the Terminal to automatically reformat code according to the Black Code Style. 

 ## Organization: things to TODO.

- Do both: 
   - training on the first half of timesteps predicting the second half, and;
   - training on the second half of timesteps and the predicting the first half.
- Look at how people structure language models (NLP). They are tested on predicting   arbitrary future timesteps. 
- Use something search based for logging.
- Think about canonical plots that you always want to make, e.g.:
      - a bunch of curves where I hold all else constant except for 1 config item.
      - each different config line a different color.
- Add unit tests for each config.
- Various tasks to implement:
   - predict the identity of the neuron given the trace (node prediction).
   - predict whether or not there exist an edge (edge prediction). 
   - predict the behavior of the worm from its neural activity.
- Goals for next week + future:
   - still don't have networks that surpass the baseline.
   - see whether we can do better with a GNN with/without connectome constraint.
    - do we need additional biological information.
   - scaling laws:
      - varying dataset sizes.
      - difference between training one one worm versus many worms.
      - change how long the dataset you use from each worm.
- tangible goal for next week
   - scaling law plots!
