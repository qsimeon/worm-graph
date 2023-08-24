# Models Submodule

This submodule contains code for defining and utilizing various models for neural data analysis.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the main script for loading the model specified in the configuration file, with its respective hyperparameters (see the configs submodule for more details).
- `_utils.py`: Contains the implementation of the models.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the configuration file `configs/submodule/model.yaml` to specify the model type, size, and other relevant parameters.
3. Run the `python main.py +submodule=model` to instantiate or load the model based on the provided configuration.
4. For more usage examples, see the configuration submodule.

**Note:** Ensure that the required checkpoints or model files are available in the specified locations if loading a saved model.

## Model Classes & Structure

The `_utils.py` script includes several model classes, which are subclasses of the `Model` superclass. Each model class represents a different type of neural network model and provides specific implementation details.

1. We have defined a superclass called `Model`, from which you can inherit to create specific model architectures.

2. This `Model` class has multiple methods and attributes, some of which include loss functions, an initialization function for hidden states, and getter functions for retrieving the attributes needed to reinstantiate a similar model.

3. The `forward()` method of the `Model` class performs the main operations of the model and outputs the final results.

4. In our `Model` class, we have also defined a `loss_fn()` method which calculates the loss with Fast Fourier Transform (FFT) regularization and L1 regularization on all model weights if requested.

5. The `generate()` method in out `Model` class is for generating future timesteps of neural activity. The output tensor gets initialized and then populated in a loop that conditions on the previous `context_len timesteps`, gets the prediction of the next timestep, and appends the predicted next timestep to the running sequence.

6. We also have a `sample()` method, but it currently isn't implemented yet.

The available model implementations (i.e. subclasses of `Model`) are:

- `LinearNN`: This model is a simple linear regression model with an embedding layer and feedforward blocks.

- `NeuralTransformer`: This model is a transformer for neural activity data. It includes a positional encoding layer, an embedding layer, and transformer blocks. This model does not use LayerNorm in its input to hidden and hidden to hidden transformations.

- `NetworkRNN`: This model is a representation of the _C. elegans_ nervous system using a continuous-time RNN backbone. It uses a LayerNorm in its input to hidden transformation.

- `NeuralCFC`: This is a Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model. It uses a LayerNorm in its input to hidden transformation and also provides methods for initializing hidden state and weights.

- `NetworkLSTM`: This model is a representation of the _C. elegans_ neural network using an LSTM. It uses a LayerNorm in its input to hidden transformation and also provides methods for initializing hidden state and weights.

- `NetworkGCN`: This is a graph neural network model of the _C. elegans_ nervous system. This model does not use a LayerNorm. The model assumes the connectome graph data to be present in a specific path and loads it.

The overarching structure of these models is the same. They all contain an input-to-hidden transformation (embedding layer + activation function + LayerNorm (optional)), and a hidden-to-hidden transformation (usually a specific neural network layer or block). Also, they all have the `InnerHiddenModel` instantiated for use.

The structure of all of the models is based on three primary components:

1. The _input_ to _hidden_ block
2. The _hidden_ to _hidden_ block
3. The _hidden_ to _output_ block

The first and last blocks act as linear maps, transforming the inputs to a latent representation (the hidden part) and transforming the latent representation to outputs, respectively. These blocks remain consistent across different network architectures, effectively serving to rescale the dimensions of the inputs/outputs. The computation within the hidden states, however, can vary. For instance, in a Linear model, the _hidden_ to _hidden_ block comprises a straightforward Feed Forward Network, while in an RNN model, it includes an RNN submodule.


## Customization

One thing to note is that the models have been designed to be flexible. The constructors allow for the setting of the input size, hidden size, the number of layers, loss functions, and regularization parameters. This design promotes reusability and modularity.

Keep in mind that not all models use all parameters (for example, the `num_layers` parameter isn't used in some models), so there might be an opportunity to refine these models further based on the specific use case or to generalize the constructor parameters in a way that's applicable to all types of models in this set.

You can select the desired model type in the configuration file `configs/submodule/model.yaml` by specifying the `type` parameter.

This design approach was chosen to facilitate customization for the end user. Users only need to implement the _hidden_ to _hidden_ block in `_utils.py` when creating a new model, thereby streamlining the process.

