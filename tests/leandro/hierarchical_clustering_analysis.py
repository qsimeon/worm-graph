import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from itertools import cycle
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

    elif group_by==None:
        for key, value in neuron_classification.items():
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
        assert len(set(ref_dict.values())) == 7, f"Invalid ref_dict -> Must have 7 unique values"

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
        hist = sns.histplot(data=new_df, x='Computed Cluster', stat=stat, discrete=True, kde=True, ax=axes[1])

        # Set the labels and title for the second subplot
        axes[1].set_title('Computed cluster labels distribution')
        axes[1].set_xlabel('Neuron type')

        # Change the xticks to the correct labels
        axes[1].set_xticks(np.arange(len(set(new_df['Computed Cluster'])))+1)

        # Compute the proportions of each Reference label within each bin
        if stat == 'percent':
            color_palette = sns.color_palette('Set1')
            unique_references = new_df['Reference'].unique()[:len(set(new_df['Computed Cluster']))]
            color_map = {ref: color_palette[i % len(color_palette)] for i, ref in enumerate(unique_references)}


            for patch in hist.patches:
                x = patch.get_x()
                width = patch.get_width()
                height = patch.get_height()
                bin_label = int(x + width / 2)  # Compute the label of the bin
                proportions = new_df[new_df['Computed Cluster'] == bin_label]['Reference'].value_counts(normalize=True)
                cumulative_height = 0
                for ref, proportion in proportions.items():
                    color = color_map.get(ref, 'gray')
                    ref_height = height * proportion
                    axes[1].bar(
                        x + width / 2, ref_height, width=width, bottom=cumulative_height,
                        color=color, label=ref, alpha=0.5, edgecolor='black'
                    )
                    cumulative_height += ref_height

            # Add legend for the first four items
            legend_elements = [Patch(facecolor=color_map.get(ref, 'gray'), edgecolor='black', label=ref)
                            for ref in new_df['Reference'].unique()[:len(set(new_df['Computed Cluster']))]]
            axes[1].legend(handles=legend_elements, loc='upper right')

        # Adjust the layout and spacing between subplots
        plt.tight_layout()

        # Show the plot
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

def create_ref_column(df, ref_dict):
    new_df = df.copy()
    new_df['Reference'] = [ref_dict[neuron] if neuron in ref_dict.keys() else np.NaN for neuron in df.index]
    return new_df

def delete_ref_column(df):
    new_df = df.copy()
    new_df = new_df.drop('Reference', axis=1)
    return new_df

def hc_analyse_dataset(dataset, apply_suggestion=False, hip='hip1', group_by='four', method='ward', metric=None):
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

    ref_dict = load_reference(group_by=group_by) # Create same ref dict for all worms

    num_worms = len(dataset.keys())
    print(f'Number of worms: {num_worms}')

    if not apply_suggestion:
        print('No suggestion applied, ignoring hip parameter.')
    else:
        print(f'Suggestion applied: {hip}.')

    # ===

    silhouettes = []
    all_worm_clusters_list = []
    count_inside_clusters_array = np.zeros((num_worms, num_clusters, groups))

    # ===

    for i, wormID in enumerate(dataset.keys()):
        clusters, silhouette_avg = hierarchical_clustering_algorithm(dataset[wormID],
                                        method=method, metric=metric, show_plots=False,
                                        criterion='maxclust', criterion_value=num_clusters,
                                        ) # Compute clusters
        
        silhouettes.append(silhouette_avg) # Save silhouette score

        grouped_clusters = neuron_distribution(clusters, ref_dict=ref_dict, group_by=group_by, show_plots=False) # Group clusters

        sugg_dict = suggest_classification(grouped_clusters) # Suggest classification

        if apply_suggestion:
            all_worm_clusters_list.append(grouped_clusters['Computed Cluster'].apply(cluster2suggestion, suggestion=sugg_dict[hip]).drop(columns=['Reference']))
        else:
            all_worm_clusters_list.append(grouped_clusters['Computed Cluster'].drop(columns=['Reference']))
        
        count_inside_clusters_array[i, :, :] = delete_total(count_inside_clusters(grouped_clusters, percentage=False)).values #? Count instead of percent?
    
    all_worm_clusters = pd.concat(all_worm_clusters_list, axis=1, keys=range(1, len(all_worm_clusters_list) + 1))
    all_worm_clusters.columns = [f"worm{i}" for i in range(0, len(all_worm_clusters_list))]

    all_worm_clusters = create_ref_column(all_worm_clusters, ref_dict) # Add reference column

    if apply_suggestion:
        # Accuracy of the classification for each worm
        for wormID in all_worm_clusters.columns[:-1]:
            # Select the wormN and reference columns
            s = all_worm_clusters[[wormID, 'Reference']].dropna()
            # Count +1 for each match between the wormN and reference columns
            s['count'] = s.apply(lambda x: 1 if x[wormID] == x['Reference'] else 0, axis=1)
            # Create row for the accuracy of the worm
            all_worm_clusters.loc['accuracy', wormID] = s['count'].sum() / len(s)

        # Accuracy of the classification for each neuron
        for neuron in all_worm_clusters.index[:-1]:
            # Compare the classifications of the neuron and compare to its reference
            s = all_worm_clusters.loc[neuron].iloc[:-1].dropna().value_counts()
            ref = all_worm_clusters.loc[neuron, 'Reference']
            # Create row for the accuracy of the neuron
            all_worm_clusters.loc[neuron, 'accuracy'] = s[ref] / s.sum()

        all_worm_clusters = delete_ref_column(all_worm_clusters) # Delete reference column
        all_worm_clusters = create_ref_column(all_worm_clusters, ref_dict) # Add reference column

    return all_worm_clusters, ref_dict, count_inside_clusters_array, silhouettes


def count_boxplot(count_inside_clusters_array):
    # Reshape the data into a 2D array
    reshaped_data = np.reshape(count_inside_clusters_array, (count_inside_clusters_array.shape[0], -1))

    # Number of neuron types/clusters
    ntypes = count_inside_clusters_array.shape[2]

    # Create a cycle colormap using the 'Set1' color palette from seaborn
    color_palette = cycle(sns.color_palette('Set1')[0:ntypes])

    if ntypes == 4:
        labels = ['I', 'M', 'P', 'S']
    elif ntypes == 3:
        labels = ['I', 'M', 'S']
    else:
        labels = ['I', 'M', 'MI', 'S', 'SI', 'SM', 'SMI']

    # Create a box plot for the reshaped data with cycle colormap
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(reshaped_data, patch_artist=True, boxprops={'facecolor': 'white'}, capprops={'color': 'black'},
                    whiskerprops={'color': 'black'}, medianprops={'color': 'black'})

    # Cycle through the colors for every 4 boxes
    for i, box in enumerate(bp['boxes']):
        color = next(color_palette)
        box.set(facecolor=color, alpha=0.5)

        if i % ntypes == 0 and i != 0:
            ax.axvline(i + 0.5, color='gray', linestyle='dashed')

    # Set the x-axis tick labels
    ax.set_xticks(np.arange((reshaped_data.shape[1]/ntypes+1)/2, reshaped_data.shape[1], ntypes))
    ax.set_xticklabels([f'Computed Cluster ({c+1})' for c in range(ntypes)])

    # Set the title and labels for the plot
    ax.title.set_text(f'Dispersion of number of Neurons per Cluster')
    ax.set_ylabel("Counts")

    # Add legend
    legend = [bp['boxes'][i] for i in range(ntypes)]
    ax.legend(legend, labels, loc='upper right')

    # Adjust the layout and spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def count_clustered_times(all_worms_clusters, neuron_names, verbose=True):
    new_df = all_worms_clusters.copy()

    # Check if all neurons are present in the DataFrame
    if all(neuron in new_df.index for neuron in neuron_names):
        # Get the values for the neurons
        values = new_df.loc[neuron_names].values

        # Count the number of times they were clustered together
        equal_values = np.all(np.equal(values, values[0]), axis=0)
        non_null_values = ~np.isnan(values).any(axis=0)
        count = np.sum(equal_values & non_null_values)
        occurrences = np.sum(non_null_values)
        freq = count / occurrences

        if verbose:
            print(f"The neurons '{neuron_names}' were clustered together {count} times (out of {occurrences}).")
    else:
        if verbose:
            print("One or more of the neurons are not present in the DataFrame.")

    return freq, count, occurrences

def group_frequency(all_worm_clusters, neuron_groups=None, ocurr_threshold=5, show_plot=True, verbose=True):

    new_df = all_worm_clusters.copy()

    if neuron_groups is None:
        neuron_groups = [[neuron] for neuron in new_df.index.to_list()]
        pair_wise = True
    else:
        pair_wise = False

    freq_matrix = np.zeros((len(neuron_groups), len(new_df.index.to_list())))
    new_groups = []

    for row, neuron1 in enumerate(neuron_groups): # neuron groups in rows
        for col, neuron2 in enumerate(new_df.index.to_list()): # all neurons in columns

            if pair_wise: # Comparing all neurons, we can take just the lower triangle
                if row > col:
                    freq, count, occurrences = count_clustered_times(new_df, neuron1 + [neuron2], verbose=False)
                    freq_matrix[row, col] = freq
                    if freq == 1 and occurrences > ocurr_threshold:
                        new_groups.append(neuron1 + [neuron2])
                    
            else: # Comparing neuron groups, we need to take all the values
                if neuron2 in neuron1:
                    freq_matrix[row, col] = 1
                    freq = 1
                    continue
                else:
                    freq, count, occurrences = count_clustered_times(new_df, neuron1 + [neuron2], verbose=False)
                    freq_matrix[row, col] = freq
                    if freq == 1 and occurrences > ocurr_threshold:
                        new_groups.append(neuron1+[neuron2])

    if verbose:
        print('Number of neuron groups: {}'.format(len(new_groups)))

    if show_plot:
        if pair_wise:
            plotHeatmap(freq_matrix, cmap='coolwarm', center=0.5, xlabel='Neuron', ylabel='Neuron',
                        xticks=new_df.index.to_list(), yticks=neuron_groups, mask='UPPER_T',
                        xtick_skip=3, ytick_skip=3, title='Neuron clustering co-occurrence frequency matrix')
        else:
            plotHeatmap(freq_matrix, cmap='coolwarm', center=0.5, xlabel='Neuron', ylabel='Neuron',
                        xticks=new_df.index.to_list(), yticks=neuron_groups,
                        xtick_skip=3, ytick_skip=2, title='Neuron clustering co-occurrence frequency matrix')
    
    return new_groups