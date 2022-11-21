# worm-graph
## Simulating worms with graph nets.

```.
├── Data
   ├── load_connectome.py
   ├── load_neural_activity.py
   ├── iter_dataset.py
   ├── map_dataset.py
   ├── batch_sampler.py
   └── PlotRealData.ipynb
├── Preprocessing
   ├── process_raw.py
   ├── PickleNeuralData.ipynb
   ├── export_nodes_edges.m
   ├── graph_tensors.pt
   ├── Nguyen2017.pickle
   ├── Kaplan2020.pickle
   └── Uzel2022.pickle
├── Models
   ├── RNN Models
   ├── GNN Models
   └── Flexible Frameworks
├── Tasks
   ├── Time-series prediction
   ├── Structure prediction
   ├── Perturbation experiments
   └── Self-supervised tasks
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
   *Important:* Do not skip the step above. Otherwise you will face a lot of `ModuleNotFoundError`s.

Verify that the new environment was installed correctly: `conda env list`
   You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

## Get started with the pipeline in 1-line

`python -i main.py`

 ## Naming conventions
 
 Folders, use lowercase letters with underscores to separate words.
 **Example:** `my_folder`.

 ## Style conventions
 * Aim to make every script not significantly longer than 100 lines. If your code is getting longer than this, it probably is a 
   good idea to modularize things by putting certain functions or classes in separare files like `utils.py` or `models.py`, etc.
 * Always shape neural data matrices as `(time, neurons, features)`.


 ## Organization: things to TODO.
- Do both: 
   - training on the first half of timesteps predicting the second half, and;
   - training on the second half of timesteps and the predicting the first half.
- Look at how people structure language models (NLP). They are tested on predicting arbitrary future timesteps. 
- Various tasks to implement:
   - predict the identity of the neuron given the trace (node prediction).
   - predict the behavior of the worm from its neural activity.
   - edge prediction: predict whether or not there exist an edge. 