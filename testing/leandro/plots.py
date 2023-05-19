import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def plot_signals(data, time_tensor, num_columns):
    assert isinstance(data, torch.Tensor), "data must be a PyTorch tensor"
    assert isinstance(time_tensor, torch.Tensor), "time_tensor must be a PyTorch tensor"
    assert data.dim() == 2, "data must be a 2D tensor"
    assert time_tensor.dim() == 1, "time_tensor must be a 1D tensor"
    assert data.size(0) == time_tensor.size(0), "Number of rows in data and time_tensor must match"
    
    num_neurons = data.size(1)
    assert num_columns <= num_neurons, "num_columns cannot exceed the number of neurons"
    
    # Randomly select the column indices
    column_indices = np.random.choice(num_neurons, num_columns, replace=False)
    
    # Extract the selected columns from the data tensor
    selected_columns = data[:, column_indices]
    
    # Define the color palette using scientific colors
    colors = sns.color_palette("bright", num_columns)
    
    # Plotting subplots vertically
    fig, axs = plt.subplots(num_columns, 1, figsize=(15,num_columns))
    fig.tight_layout(pad=0.0)
    
    for i, ax in enumerate(axs):
        ax.plot(time_tensor, selected_columns[:, i], color=colors[i])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Neuron {}".format(column_indices[i]))
        
        if i < num_columns - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time")
    
    plt.show()

def compare_signals(data1, data2, x, k):
    # Ensure data1 and data2 are torch tensors
    if not isinstance(data1, torch.Tensor):
        data1 = torch.Tensor(data1)
    if not isinstance(data2, torch.Tensor):
        data2 = torch.Tensor(data2)

    # Ensure x is a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # Check dimensions
    assert data1.shape == data2.shape, "data1 and data2 must have the same shape"
    assert len(data1.shape) == 2, "data1 and data2 must be 2D"
    assert x.shape[0] == data1.shape[0], "x must have the same number of rows as data1 and data2"
    assert k <= data1.shape[1], "k must be less than or equal to the number of columns in data1 and data2"

    # Randomly select k column indices
    indices = torch.randperm(data1.shape[1])[:k]

    # Create a grid spec plot with 2 rows for each index
    fig = plt.figure(figsize=(15, int(2.5*k)))
    gs = gridspec.GridSpec(2*k, 1)

    # Define a color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # For each index, plot the corresponding columns from data1 and data2 in separate subplots
    for i, idx in enumerate(indices):
        color = colors[i % len(colors)]  # cycle colors

        ax1 = fig.add_subplot(gs[2*i, 0])
        ax1.plot(x, data1[:, idx].numpy(), label='data1', color=color)
        ax1.legend()
        ax1.set_ylim(-1.0, 1.0)
        ax1.xaxis.set_ticks([])  # remove x-axis ticks
        ax1.spines['right'].set_visible(False)  # remove right border
        ax1.spines['top'].set_visible(False)  # remove top border
        ax1.spines['bottom'].set_visible(False)  # remove bottom border

        # Set y label for the outermost plot
        ax1.set_ylabel(f'Neuron {idx} A', rotation=45, labelpad=40, verticalalignment='center', fontsize=12)

        ax2 = fig.add_subplot(gs[2*i + 1, 0])
        ax2.plot(x, data2[:, idx].numpy(), label='data2', color=color)
        ax2.legend()
        ax2.set_ylim(-1.0, 1.0)
        ax2.spines['right'].set_visible(False)  # remove right border
        ax2.spines['top'].set_visible(False)  # remove top border

        # Set y label for the outermost plot
        ax2.set_ylabel(f'Neuron {idx} B', rotation=45, labelpad=40, verticalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()


