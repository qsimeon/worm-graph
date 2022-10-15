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

 ## Style conventions (TODO)
 Aim to make every script no more than 100 lines. If your code is getting longer than this, it probably is a good idea to modularize thnings by putting certain functions or classes in separare files like `utils.py` or `models.py`, etc.


 ## Organization: things to TODO.
- Move the load of the preprocessing to a separate preprocessing file rather than `data_loader.py.`
- Add docstrings to the models files. Add docstrings to all modules (classes and functions) existing and those made in the future.
- Do both training on the first half timesteps predicting second half.
   - as well as training second half and predicting first.
- Look at how do people structure language models (NLP). Tested on predicting arbitrary future timesteps. How to structure the code effectively. 
   - Instead of pre-slicing the data, just have a function that returns you a batch like `get_batch`, and then you can arbitrarily set the values.
      - essentially you want a generator function.
      - set the dataset as a queue.
      - set one-step prediction as a special case of n-step prediction.
- predict the idenetity of the neuron given the trace (node prediction).
- predict the behavior of the worm?, what is the action, what is the shape of the body?
- edge prediction, predict whether or not there exist an edge. if we know the identity of the neurons the we know the target.


