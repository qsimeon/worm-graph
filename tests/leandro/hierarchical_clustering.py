from tests.leandro.tools import correlation_matrix
from tests.leandro.plots import plotHeatmap

import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate
import json

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

def hierarchical_clustering_analysis(single_worm_data,
                                     method='complete', metric='euclidean',
                                     truncate_mode='lastp', p=12, plot_dendrogram=True,
                                     criterion='maxclust', criterion_value=4, verbose=False):
    """
        single_worm_data: single worm dataset
        method: linkage method
        metric: distance metric
        plot: whether to plot the dendrogram or not
    """

    np.set_printoptions(precision=4, suppress=True)
    plt.figure(figsize=(10, 3))
    plt.style.use('seaborn-whitegrid')

    X = single_worm_data['smooth_calcium_data'] # (time, all neurons)
    X = X[:, single_worm_data['named_neurons_mask']]  # (time, named and acive neurons)

    R = correlation_matrix(X)
    R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
    D = 1 - R # Distance matrix
    np.fill_diagonal(D, 0) # Make diagonal 0 (just in case)

    if verbose:
        print("X.shape:", X.shape)
        print("Correlation matrix shape:", R.shape)

    # The linkage function takes a condensed distance matrix, which is a flat array containing the upper triangular of the distance matrix. 
    # We use squareform function to convert the matrix form to the condensed form.
    condensed_D = squareform(D)
    Z = linkage(condensed_D, method=method, metric=metric)

    dendrogram(Z, truncate_mode=truncate_mode, p=p, leaf_rotation=45., leaf_font_size=10., show_contracted=True)

    if plot_dendrogram:
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        plt.show()

    computed_cluster_labels = fcluster(Z, criterion_value, criterion=criterion)

    # === Sorting ===
    original_neuron_labels = np.array([label for idx, label in single_worm_data['slot_to_named_neuron'].items()])

    # Now we can sort the correlation matrix according to the cluster labels, and plot the correlation matrix again.
    sorted_R = R[:, np.argsort(computed_cluster_labels)] # sort columns
    sorted_R = sorted_R[np.argsort(computed_cluster_labels), :] # sort rows
    sorted_neuron_labels = original_neuron_labels[np.argsort(computed_cluster_labels)]
    sorted_computed_cluster_labels = computed_cluster_labels[np.argsort(computed_cluster_labels)]

    plotHeatmap(R, title="Original correlation matrix", xlabel="Neuron", ylabel="Neuron", xticks=original_neuron_labels, yticks=original_neuron_labels, xtick_skip=2, ytick_skip=2)
    plotHeatmap(sorted_R, title="Sorted correlation matrix", xlabel="Neuron", ylabel="Neuron", xticks=sorted_neuron_labels, yticks=sorted_neuron_labels, xtick_skip=2, ytick_skip=2)

    # === Metrics ===
    file_path = '/home/lrvnc/Projects/worm-graph/tests/leandro/data/neuron_clusters.json'

    try:
        with open(file_path, 'r') as f:
            neuron_classification = json.load(f)
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error while decoding JSON: {e}")

    clusters = {}
    for idx, neuron in enumerate(sorted_neuron_labels):
        clusters[neuron] = {'Computed Cluster': sorted_computed_cluster_labels[idx], 'Reference': ', '.join(neuron_classification[neuron])}

    clusters = pd.DataFrame.from_dict(clusters, orient='index')
    count_inside_clusters = clusters.groupby('Computed Cluster')['Reference'].value_counts().unstack().fillna(0)

    return clusters, count_inside_clusters




