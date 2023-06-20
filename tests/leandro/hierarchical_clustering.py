from tests.leandro.plots import plotHeatmap

import numpy as np
import pandas as pd
import json

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform, euclidean
from dtaidistance import dtw

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

def hierarchical_clustering_algorithm(single_worm_data, distance='correlation',
                                     method='ward', metric=None,
                                     truncate_mode='lastp', p=12,
                                     criterion='maxclust', criterion_value=4, verbose=False,
                                     show_plots=True):
    """
        single_worm_data: single worm dataset
        method: linkage method
        metric: distance metric
        plot: whether to plot the dendrogram or not
    """

    np.set_printoptions(precision=4, suppress=True)
    if show_plots:
        plt.figure(figsize=(10, 3))
        plt.style.use('seaborn-whitegrid')

    X = single_worm_data['smooth_calcium_data'] # (time, all neurons)
    X = X[:, single_worm_data['named_neurons_mask']]  # (time, named and acive neurons)

    if distance == 'correlation':
        R = np.abs(np.corrcoef(X, rowvar=False)) # no correlated <- [0, 1] -> correlated
        R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
        D = 1 - R # Distance matrix: close <- [0, 1] -> far
        np.fill_diagonal(D, 0) # Make diagonal 0 (just in case)
        title_plot = "correlation matrix"
        
    elif distance == 'cosine':
        R = np.abs(X.T @ X) # Distance matrix: far <- [0, 1] -> close
        norms = np.linalg.norm(X, axis=0).reshape(-1,1) @ np.linalg.norm(X, axis=0).reshape(1,-1)
        R = (R / norms).detach().numpy() # cosine similarity
        R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
        D = 1 - R # Distance matrix: close <- [0, 1] -> far
        np.fill_diagonal(D, 0) # Make diagonal 0 (just in case)
        title_plot = "cosine similarity matrix"

    elif distance == 'dtw':
        X = X.detach().numpy().astype(np.double).T
        time_series = [vec for vec in X]
        R = dtw.distance_matrix_fast(time_series, window=1000) # window: anti-phase dynamics around 50s and 100s
        D = R
        title_plot = "DTW matrix"

    if verbose:
            print("X.shape:", X.shape)
            print("Distance matrix shape:", D.shape)

    # The linkage function takes a condensed distance matrix, which is a flat array containing the upper triangular of the distance matrix. 
    # We use squareform function to convert the matrix form to the condensed form.
    condensed_D = squareform(D)
    Z = linkage(condensed_D, method=method, metric=metric)

    # === Plot dendrogram ===
    if show_plots:
        dendrogram(Z, truncate_mode=truncate_mode, p=p, leaf_rotation=45., leaf_font_size=10., show_contracted=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        plt.show()

    # === Cluster labels ===
    computed_cluster_labels = fcluster(Z, criterion_value, criterion=criterion)
    silhouette_avg = sm.silhouette_score(D, computed_cluster_labels, metric='cosine') # Quality of the clustering -> cosine distance gave best results


    # === Sorting ===
    original_neuron_labels = np.array([label for idx, label in single_worm_data['slot_to_named_neuron'].items()])

    # Now we can sort the correlation matrix according to the cluster labels, and plot the correlation matrix again.
    sorted_R = R[:, np.argsort(computed_cluster_labels)] # sort columns
    sorted_R = sorted_R[np.argsort(computed_cluster_labels), :] # sort rows
    sorted_neuron_labels = original_neuron_labels[np.argsort(computed_cluster_labels)]
    sorted_computed_cluster_labels = computed_cluster_labels[np.argsort(computed_cluster_labels)]

    if show_plots:
        plotHeatmap(R, title="Original " + title_plot, xlabel="Neuron", ylabel="Neuron", xticks=original_neuron_labels, yticks=original_neuron_labels, xtick_skip=2, ytick_skip=2)
        plotHeatmap(sorted_R, title="Sorted " + title_plot, xlabel="Neuron", ylabel="Neuron", xticks=sorted_neuron_labels, yticks=sorted_neuron_labels, xtick_skip=2, ytick_skip=2)

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

    # Define the replacements
    replacements = {
        'interneuron': 'I',
        'motor': 'M',
        'sensory': 'S',
        'motor, interneuron': 'MI',
        'sensory, motor': 'SM',
        'sensory, interneuron': 'SI',
        'sensory, motor, interneuron': 'SMI',
        'polymodal': 'P'
    }

    # Replace the values in the 'Reference' column
    clusters['Reference'] = clusters['Reference'].replace(replacements)

    clusters.index.name = 'Neuron'

    return clusters, silhouette_avg