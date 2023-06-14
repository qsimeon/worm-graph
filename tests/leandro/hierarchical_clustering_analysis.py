import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .hierarchical_clustering import *

def load_reference(group_by=None):
    file_path = '/home/lrvnc/Projects/worm-graph/tests/leandro/data/neuron_clusters.json'

    try:
        with open(file_path, 'r') as f:
            neuron_classification = json.load(f)
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error while decoding JSON: {e}")

    replacements = {
        'interneuron': 'I',
        'motor': 'M',
        'sensory': 'S',
        'motor, interneuron': 'MI',
        'sensory, motor': 'SM',
        'sensory, interneuron': 'SI',
        'sensory, motor, interneuron': 'SMI',
        'unknown': 'U',
    }

    for key, value in neuron_classification.items():
            text = ', '.join(neuron_classification[key])
            neuron_classification[key] = replacements[text]

    if group_by=='four':
        for key, value in neuron_classification.items():
            if value == 'MI' or value == 'SM' or value == 'SI' or value == 'SMI':
                neuron_classification[key] = 'P'

            if value == 'U':
                neuron_classification[key] = np.random.choice(['M', 'I', 'S'])
    
    elif group_by=='three':
        
        for key, value in neuron_classification.items():
            if value == 'MI' or value == 'SM' or value == 'SI' or value == 'SMI':
                neuron_classification[key] = np.random.choice([char for char in value])

            if value == 'U':
                neuron_classification[key] = np.random.choice(['M', 'I', 'S'])

    return neuron_classification

def random_replace(value):
    # Aux function to randomly replace the values with equal distribution
    if len(value) > 1:
        possible_labels = [char for char in value]
        return np.random.choice(possible_labels)
    else:
        return value

def neuron_distribution(df, stat='percent', group_by=None, show_plots=True):

    assert group_by in [None, 'three', 'four'],\
        f"Invalid group_by: {group_by} -> Must be None, 'three' or 'four'"

    assert stat in ['percent', 'count', 'proportion', 'density'], \
        f"Invalid stat: {stat} -> Must be 'percent', 'count', 'proportion' or 'density'"

    new_df = df.copy()

    # Group the neurons by three or four
    if group_by == 'four':
        new_df.loc[new_df['Reference'].str.len() > 1, 'Reference'] = 'P'
    elif group_by == 'three':
        new_df['Reference'] = new_df['Reference'].apply(random_replace)

    if show_plots:
        # Create the histogram
        sns.histplot(data=new_df, x='Reference', stat=stat, discrete=True, kde=True)

        # Set the labels and title
        plt.title(f'Neuron distribution ({stat})')
        plt.xlabel('Neuron type')

        # Display the plot
        plt.tight_layout()
        plt.show()

    return new_df

def create_total(df):
    new_df = df.copy()
    new_df.loc['total'] = new_df.sum(axis=0) # Count over columns
    new_df['total'] = new_df.sum(axis=1) # Count over rows
    return new_df

def delete_total(df):
    new_df = df.copy()
    new_df = new_df.drop('total', axis=0) # Drop row
    new_df = new_df.drop('total', axis=1) # Drop column
    return new_df

def count_inside_clusters(df, stat='count', dimension='reference'):

    assert stat in ['count', 'percent'],\
        f"Invalid stat: {stat} -> Must be 'count' or 'percent'"
    
    new_df = df.copy()

    new_df = new_df.groupby('Computed Cluster')['Reference'].value_counts().unstack().fillna(0)

    new_df = new_df.astype(int)
    new_df = create_total(new_df)

    if stat == 'percent':
        new_df = convert_to_percentages(new_df, dimension=dimension)

    return new_df

def convert_to_percentages(df, dimension='reference'):
    assert dimension in ['reference', 'computed-cluster'], f"Invalid dimension: {dimension} -> Must be 'reference' or 'computed-cluster'"

    new_df = df.copy()

    # create total row and column if they don't exist
    if 'total' not in new_df.index:
        new_df = create_total(new_df)
    
    if dimension == 'reference':
        new_df = new_df.div(new_df.loc['total'], axis=1)*100
    elif dimension == 'computed-cluster':
        new_df = new_df.div(new_df['total'], axis=0)*100

    return new_df.round(decimals=2)

def suggest_classification(grouped_clusters):

    # Suggestion matrix: (0, :, :) -> suggestion inside cluster (computed-cluster dimension) -> max: left-right
    #                    (1, :, :) -> global suggestion (reference dimension) -> max: top-down

    new_df = grouped_clusters.copy()

    count_df = delete_total(count_inside_clusters(new_df, stat='percent', dimension='reference'))

    suggestion_matrix = np.empty((2, count_df.shape[0], count_df.shape[1]), dtype=object)

    # Suggestion inside cluster (computed-cluster dimension)
    for i, _ in enumerate(count_df.index):
        count_df = delete_total(count_inside_clusters(new_df, stat='percent', dimension='computed-cluster'))
        suggestion_matrix[0, i, :] = count_df.columns[np.argsort(count_df.loc[i+1].values)].values

    # Global suggestion (reference dimension)
    for j, col_name in enumerate(count_df.columns):
        count_df = delete_total(count_inside_clusters(new_df, stat='percent', dimension='reference'))
        suggestion_matrix[1, :, j] = count_df.index[np.argsort(count_df[col_name].values)].values

    suggestion_inside_cluster = {i+1: c for i, c in enumerate(suggestion_matrix[0, :, -1])}
    
    suggestion_global = {i+1: list() for i in range(suggestion_matrix.shape[1])}
    for i, col_name in enumerate(count_df.columns):
        suggestion_global[suggestion_matrix[1, -1, i]].append(col_name)

    for key, item in suggestion_global.items():
        if len(item) == 0:
            suggestion_global[key] = np.nan
        else:
            suggestion_global[key] = ''.join(item)
    
    return suggestion_inside_cluster, suggestion_global

def cluster2suggestion(value, suggestion):
    return suggestion[value]

def accuracy_by_neuron():
    pass

def accuracy_by_worm():
    pass

def analyse_dataset(dataset, group_by='four', method='ward', metric=None, stat='percent'):
    """
        dataset = loaded dataset
    """

    if group_by == 'four':
        groups = 4
        num_clusters = 4
    elif group_by == 'three':
        groups = 3
        num_clusters = 3
    else:
        groups = 7
        num_clusters = 7

    num_worms = len(dataset.keys())
    print(f'Number of worms: {num_worms}')

    # ===

    silhouettes = []
    all_worm_clusters_list = [[], []] # [[suggestion 1], [suggestion 2]]
    count_inside_clusters_array = np.zeros((num_worms, num_clusters, groups))

    # ===

    for i, wormID in enumerate(dataset.keys()):
        clusters, silhouette_avg = hierarchical_clustering_algorithm(dataset[wormID],
                                        method=method, metric=metric, show_plots=False,
                                        criterion='maxclust', criterion_value=num_clusters,
                                        )

        grouped_clusters = neuron_distribution(clusters, stat=stat, group_by=group_by, show_plots=False)

        s1, s2 = suggest_classification(grouped_clusters=grouped_clusters)

        all_worm_clusters_list[0].append(grouped_clusters['Computed Cluster'].apply(cluster2suggestion, suggestion=s1).drop(columns=['Reference']))
        all_worm_clusters_list[1].append(grouped_clusters['Computed Cluster'].apply(cluster2suggestion, suggestion=s2).drop(columns=['Reference']))
        silhouettes.append(silhouette_avg)

        count_inside_clusters_array[i, :, :] = delete_total(count_inside_clusters(grouped_clusters, stat='count')).values

