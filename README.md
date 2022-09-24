# worm-graph
Simulating worms with graph nets

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
 
 
 Naming conventions
 
 Folders, use lowercase letters with underscores to separate words.
 **Example:** `my_folder`.
 
 If you would like folders to be ordered, prefix the folder name with `_#_` where `#` represents the desired rank of that folder.
 **Example:** `_1_chordata`, `_2_vertebrata`, `_3_mammals`