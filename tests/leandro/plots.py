import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from utils import NEURONS_302

import matplotlib.animation as animation
from IPython.display import HTML

def plot_signals(data, time_tensor, neuron_idx=None, yax_limit=True):
    assert isinstance(data, torch.Tensor), "data must be a PyTorch tensor"
    assert isinstance(time_tensor, torch.Tensor), "time_tensor must be a PyTorch tensor"
    assert data.dim() == 2, "data must be a 2D tensor"
    assert isinstance(neuron_idx, (int, list)), "neuron_idx must be an integer or list"

    time_tensor = time_tensor.squeeze()
    assert data.size(0) == time_tensor.size(0), "Number of rows in data and time_tensor must match"
    
    num_neurons = data.size(1)
    
    # Randomly select the column indices if not provided
    if isinstance(neuron_idx, int):
        assert neuron_idx <= num_neurons, "neuron_idx cannot exceed the number of neurons"
        column_indices = np.random.choice(num_neurons, neuron_idx, replace=False)
    elif isinstance(neuron_idx, list):
        assert len(neuron_idx) <= num_neurons, "neuron_idx cannot exceed the number of neurons"
        column_indices = np.array(neuron_idx)

    num_columns = len(column_indices)
    
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
        if yax_limit:
            ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("{}".format(NEURONS_302[column_indices[i]]))
        
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
    # Randomly select the column indices if not provided
    if isinstance(k, int):
        assert k <= data1.shape[0], "k cannot exceed the number of neurons"
        indices = torch.randperm(data1.shape[1])[:k]
    elif isinstance(k, list):
        assert len(k) <= data1.shape[0], "k cannot exceed the number of neurons"
        indices = torch.from_numpy(np.array(k))
        k = len(k)

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
        #ax1.legend()
        ax1.set_ylim(-1.0, 1.0)
        ax1.xaxis.set_ticks([])  # remove x-axis ticks
        ax1.spines['right'].set_visible(False)  # remove right border
        ax1.spines['top'].set_visible(False)  # remove top border
        ax1.spines['bottom'].set_visible(False)  # remove bottom border

        # Set y label for the outermost plot
        ax1.set_ylabel(f'Neuron {idx} A', rotation=45, labelpad=40, verticalalignment='center', fontsize=12)

        ax2 = fig.add_subplot(gs[2*i + 1, 0])
        ax2.plot(x, data2[:, idx].numpy(), label='data2', color=color)
        #ax2.legend()
        ax2.set_ylim(-1.0, 1.0)
        ax2.spines['right'].set_visible(False)  # remove right border
        ax2.spines['top'].set_visible(False)  # remove top border

        # Set y label for the outermost plot
        ax2.set_ylabel(f'Neuron {idx} B', rotation=45, labelpad=40, verticalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_similarities(data):
    # Find the indices of the active neurons
    active_neurons = np.where(data != 0)[0]

    # Extract the data for the active neurons
    active_data = data[active_neurons]

    # Create a bar plot of the active data
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(active_neurons)), active_data.flatten())

    # Remove the top and right contours
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Set the x label
    plt.xlabel('Neuron Index')

    # Show the plot
    plt.show()


def plotHeatmap(matrix, title=None, cmap=None, xlabel=None, ylabel=None,
                xticks=None, yticks=None, show_xticks=True, show_yticks=True,
                xtick_skip=None, ytick_skip=None,
                center=None, vmin=None, vmax=None, mask=None):
    
    # Generate the mask
    if type(mask) is str:
        if mask.upper() == 'UPPER_T':
            mask = np.triu(np.ones_like(matrix))
        elif mask.upper() == 'LOWER_T':
            mask = np.tril(np.ones_like(matrix))

    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate the heatmap
    ax = sns.heatmap(matrix, mask=mask, cmap=cmap, center=center, vmin=vmin, vmax=vmax)
    
    # Set the title if provided
    if title is not None:
        ax.set_title(title)

    # Set the x and y ticks if provided and show_ticks is True
    if show_xticks:
        if xticks is not None:
            ax.set_xticks(np.arange(len(xticks))[::xtick_skip])
            ax.set_xticklabels(xticks[::xtick_skip], fontsize='small', rotation=90)
    else:
        ax.set_xticks([])

    if show_yticks:
        if yticks is not None:
            ax.set_yticks(np.arange(len(yticks))[::ytick_skip])
            ax.set_yticklabels(yticks[::ytick_skip], fontsize='small', rotation=0)
    else:
        ax.set_yticks([])

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Show the plot
    plt.tight_layout()
    plt.show()

def dynamicHeatmap(matrices, interval=200, filename='animation.mp4',
                       title=None, cmap=None,  mask=None, xlabel=None, ylabel=None, xticks=None, yticks=None,
                       show_xticks=True, show_yticks=True, center=None, vmin=None, vmax=None):
    
    
    # Generate the mask
    if type(mask) is str:
        if mask.upper() == 'UPPER_T':
            mask = np.triu(np.ones_like(matrices[0]))
        elif mask.upper() == 'LOWER_T':
            mask = np.tril(np.ones_like(matrices[0]))

    # Create the figure and axes
    fig, ax = plt.subplots()
    
    def update(frame):

        # clear the current axes
        plt.clf()
        ax = sns.heatmap(matrices[frame], mask=mask, cmap=cmap, center=center, vmin=vmin, vmax=vmax)
        
        # Set the title if provided
        if title is not None:
            ax.set_title(title)

        # Set the x and y ticks if provided and show_ticks is True
        if show_xticks:
            if xticks is not None:
                ax.set_xticks(np.arange(len(xticks)))
                ax.set_xticklabels(xticks, fontsize='small', rotation=45)
        else:
            ax.set_xticks([])

        if show_yticks:
            if yticks is not None:
                ax.set_yticks(np.arange(len(yticks)))
                ax.set_yticklabels(yticks, fontsize='small', rotation=45)
        else:
            ax.set_yticks([])

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=range(len(matrices)), interval=interval, blit=False)
    
    ani.save(filename)
    #return HTML(ani.to_html5_video())