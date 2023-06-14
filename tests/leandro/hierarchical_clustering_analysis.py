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

def neuron_distribution(df, ref_dict, stat='percent', group_by=None, show_plots=True):

    assert group_by in [None, 'three', 'four'],\
        f"Invalid group_by: {group_by} -> Must be None, 'three' or 'four'"

    assert stat in ['percent', 'count', 'proportion', 'density'], \
        f"Invalid stat: {stat} -> Must be 'percent', 'count', 'proportion' or 'density'"
    
    new_df = df.copy()

    if group_by == 'four':
        # assert just 4 unique keys in ref_dict
        assert len(set(ref_dict.values())) == 4, f"Invalid ref_dict -> Must have 4 unique values"
    elif group_by == 'three':
        # assert just 3 unique keys in ref_dict
        assert len(set(ref_dict.values())) == 3, f"Invalid ref_dict -> Must have 3 unique values"
    else:
        # assert just 8 unique keys in ref_dict
        assert len(set(ref_dict.values())) == 8, f"Invalid ref_dict -> Must have 8 unique values"

    # Replace all references by the ref_dict ones (Group the neurons by three or four)
    for neuron in new_df.index:
        new_df.loc[neuron, 'Reference'] = ref_dict[neuron]

    if show_plots:

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Create the histogram (literature)
        sns.histplot(data=new_df, x='Reference', stat=stat, discrete=True, kde=True, ax=axes[0])

        # Set the labels and title for the first subplot
        axes[0].set_title('Literature labels distribution')
        axes[0].set_xlabel('Neuron type')

        # Create the histogram (computed clusters)
        sns.histplot(data=new_df, x='Computed Cluster', stat=stat, discrete=True, kde=True, ax=axes[1])

        # Set the labels and title for the second subplot
        axes[1].set_title('Computed cluster labels distribution')
        axes[1].set_xlabel('Neuron type')
        # Set xticks
        axes[1].set_xticks(np.arange(len(set(new_df['Computed Cluster']))+1))
        axes[1].set_ylabel('')

        # Adjust the layout and spacing between subplots
        plt.tight_layout()

        # Display the plot
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

def count_inside_clusters(df, percentage=False, dimension='reference'):
    
    new_df = df.copy()

    new_df = new_df.groupby('Computed Cluster')['Reference'].value_counts().unstack().fillna(0)

    new_df = new_df.astype(int)
    new_df = create_total(new_df)

    if percentage:
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

def suggest_classification(computed_clusters_df):

    # TODO: suggestion 2

    new_df = computed_clusters_df.copy()

    # Index of the max values per columns
    count_df = delete_total(count_inside_clusters(new_df, percentage=True, dimension='reference'))
    max_values_col = count_df.idxmax(axis=0)

    # Column of the max value per row
    count_df = delete_total(count_inside_clusters(new_df, percentage=True, dimension='computed-cluster'))
    max_values_row = count_df.idxmax(axis=1)

    # Create mapping
    suggestion = {'hip1': {key: value for key, value in max_values_row.items()}, 
                  'hip2': {}} # hip1: inside cluster, hip2: global
    
    return suggestion

def cluster2suggestion(value, suggestion):
    return suggestion[value]

def accuracy_by_neuron():
    pass

def accuracy_by_worm():
    pass

def analyse_dataset(dataset, ref_dict, group_by='four', method='ward', metric=None, stat='percent'):
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

        grouped_clusters = neuron_distribution(clusters, ref_dict=ref_dict, group_by=group_by, show_plots=False)

        s1, s2 = suggest_classification(grouped_clusters=grouped_clusters)

        all_worm_clusters_list[0].append(grouped_clusters['Computed Cluster'].apply(cluster2suggestion, suggestion=s1).drop(columns=['Reference']))
        all_worm_clusters_list[1].append(grouped_clusters['Computed Cluster'].apply(cluster2suggestion, suggestion=s2).drop(columns=['Reference']))
        silhouettes.append(silhouette_avg)

        count_inside_clusters_array[i, :, :] = delete_total(count_inside_clusters(grouped_clusters, stat='count')).values

