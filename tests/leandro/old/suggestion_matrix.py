# OLD CODE (backup)

def suggest_classification(grouped_clusters):

    # Suggestion matrix: (0, :, :) -> suggestion inside cluster (computed-cluster dimension) -> max: left-right
    #                    (1, :, :) -> global suggestion (reference dimension) -> max: top-down

    new_df = grouped_clusters.copy()

    count_df = delete_total(count_inside_clusters(new_df, percentage=True, dimension='reference'))

    suggestion_matrix = np.empty((2, count_df.shape[0], count_df.shape[1]), dtype=object)

    # Suggestion inside cluster (computed-cluster dimension)
    for i, _ in enumerate(count_df.index):
        count_df = delete_total(count_inside_clusters(new_df, percentage=True, dimension='computed-cluster'))
        suggestion_matrix[0, i, :] = count_df.columns[np.argsort(count_df.loc[i+1].values)].values

    # Global suggestion (reference dimension)
    for j, col_name in enumerate(count_df.columns):
        count_df = delete_total(count_inside_clusters(new_df, percentage=True, dimension='reference'))
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