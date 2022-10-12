# worm-graph
## Simulating worms with graph nets

```.
├── Data
   ├── data_loader.py
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

`cd` into the `worm-graph` dirrectory on your local machine: `cd worm-graph`

Using the terminal or an Anaconda Prompt: `conda env create -f environment.yml`
   The first line of the `yml` file sets the new environment's name.

Activate the new environment: `conda activate worm-graph`

Add the `worm-graph` root directory to Anaconda path: `conda develop .`

Verify that the new environment was installed correctly: `conda env list`
   You can also use `conda info --envs`.
 
Always activate the environment before working on the project: `conda activate worm-graph`

 ## Naming conventions
 
 Folders, use lowercase letters with underscores to separate words.
 **Example:** `my_folder`.