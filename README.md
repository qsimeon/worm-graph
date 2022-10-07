# worm-graph
## Simulating worms with graph nets

```.
├── Data
   ├── simulatedData.py
   └── realData.py
├── Preprocessing
   ├── dataloader.py
   └── RNNdataloader.py
├── Models
   ├── RNN Models
   ├── GNN Models
   ├── Hybrid Models
   └── Flexible Frameworks
├── Tasks
   ├── Time-series prediction
   ├── Structure prediction
   ├── Perturbation experiments
   └── Supervised versus unsupervised tasks?
├── Visualizations
   ├── Connection graph
   ├── Existing voltage time series
   ├── Loss
   └── Time series predictions
└── pipelineExplorer.ipynb: Notebook to run through the pipeline, with choices (e.g. data type, model type, etc) at each step.
 ```
 
## Create the environment from the `environment.yml` file

# TODO: This is not working for Robert, please fix
Using the terminal or an Anaconda Prompt: `conda env create -f environment.yml`
   The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Verify that the new environment was installed correctly: `conda env list`
   You can also use `conda info --envs`.
 
With the `worm-graph` environment activated, install PyG: `conda install pyg -c pyg`

 ## Naming conventions
 
 Folders, use lowercase letters with underscores to separate words.
 **Example:** `my_folder`.