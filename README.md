# worm-graph
Simulating worms with graph nets


```.
├── InitialWorkingExamples
   ├── functions
   │   └── neural_interactome.py: Functions to generate activity, verbatim from Neural Interactome. We may have found discrepancy with the paper's equations here, and are completing a corrected version.
   ├── data
   │   ├── emask_default.npy: (emask * -48) is the "directionality" of each neuron (0 if excitatory or −48 mV if inhibitory), verbatim from Neural Interactome.
   │   ├── Gg_default.npy: , Gg[i,j] is the total conductivity of gap junctions between neurons i and j, verbatim from Neural Interactome.
   │   ├── Gs_default.npy: , Gs[i,j] is the total conductivity of synapses to i from j, verbatim from Neural Interactome.
   │   ├── neuron_names.txt: Names of the neurons in order, verbatim from Neural Interactome.
   │   ├── neuron_names_cleaned.txt: Names of the neurons in order, cleaned into a list.
   │   └── saved_dynamics_array.csv: Activity dynamics generated from GenerateRecordings.ipynb.
   ├── GenerateRecordings.ipynb: Generates activity dynamics randomly according to user specification, displays them.
   ├── dataset.ipynb: Generates batch dataset from 'saved_dynamics_array.csv' and connection graph, with specified adjacent timesteps to be considered for each prediction.
   ├── dataset.py: .py version of the above for loading.
   ├── LinkPredictionFromJustActivity.ipynb: Attempts to predict causal links from the dataset, displays accuracy vs ground truth.
   └── ActivityPredictionFromJustActivity.ipynb: Attempts to predict activity timeseries after a certain point, displays prediction.
 ```