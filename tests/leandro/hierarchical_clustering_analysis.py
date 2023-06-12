import pandas as pd

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

def convert_to_percentages(df, decimals=2, dimension='reference'):
    assert dimension in ['reference', 'computed-cluster'], f"Invalid dimension: {dimension} -> Must be 'reference' or 'computed-cluster'"

    new_df = df.copy()

    # create total row and column if they don't exist
    if 'total' not in new_df.index:
        new_df = create_total(new_df)
    
    if dimension == 'reference':
        new_df = new_df.div(new_df.loc['total'], axis=1)*100
    elif dimension == 'computed-cluster':
        new_df = new_df.div(new_df['total'], axis=0)*100

    return new_df.round(decimals=decimals)

def group_by_three(df):
    # Rearrange columns to three groups -> interneuron, motor, sensory
    new_df = df.copy()

    # If df has a 'total' row, drop it
    if 'total' in df.index:
        new_df = delete_total(new_df)

    # Let's consider the hypothesis that during the time that the worms were registered, the polymodal neurons had a single function (motor, interneuron or sensory).
    # Therefore, inside a computed cluster, for each combination of (motor, interneuron, sensory), we will look to the greatest count and consider that the polymodal
    # neuron belongs to that category.

    for cluster in new_df.index:

        # motor, interneuron
        max_category = df.loc[cluster][['interneuron', 'motor']].idxmax(axis=0)
        if max_category == 'motor':
            new_df.loc[cluster]['motor'] += new_df.loc[cluster]['motor, interneuron']
        elif max_category == 'interneuron':
            new_df.loc[cluster]['interneuron'] += new_df.loc[cluster]['motor, interneuron']

        # sensory, interneuron
        max_category = df.loc[cluster][['interneuron', 'sensory']].idxmax(axis=0)
        if max_category == 'sensory':
            new_df.loc[cluster]['sensory'] += new_df.loc[cluster]['sensory, interneuron']
        elif max_category == 'interneuron':
            new_df.loc[cluster]['interneuron'] += new_df.loc[cluster]['sensory, interneuron']

        # sensory, motor
        max_category = df.loc[cluster][['motor', 'sensory']].idxmax(axis=0)
        if max_category == 'sensory':
            new_df.loc[cluster]['sensory'] += new_df.loc[cluster]['sensory, motor']
        elif max_category == 'motor':
            new_df.loc[cluster]['motor'] += new_df.loc[cluster]['sensory, motor']

        # sensory, motor, interneuron
        max_category = df.loc[cluster][['interneuron', 'motor', 'sensory']].idxmax(axis=0)
        if max_category == 'sensory':
            new_df.loc[cluster]['sensory'] += new_df.loc[cluster]['sensory, motor, interneuron']
        elif max_category == 'motor':
            new_df.loc[cluster]['motor'] += new_df.loc[cluster]['sensory, motor, interneuron']
        elif max_category == 'interneuron':
            new_df.loc[cluster]['interneuron'] += new_df.loc[cluster]['sensory, motor, interneuron']

    new_df = new_df.drop(['motor, interneuron', 'sensory, interneuron', 'sensory, motor', 'sensory, motor, interneuron'], axis=1)
    new_df = create_total(new_df)

    return new_df

def group_by_four(df):
    # Rearrange columns to four groups -> interneuron, motor, sensory, polymodal

    new_df = df.copy()

    # create total column and row if not exists
    if 'total' not in df.index:
        new_df = create_total(new_df)

    new_df['polymodal'] = 0

    for col in new_df.columns:
        if ',' in col:
            new_df['polymodal'] += new_df[col]

    # Drop the columns with ',' in the name
    new_df = new_df.drop(new_df.columns[new_df.columns.str.contains(',')], axis=1)

    # Reorder the columns so total is the last one
    new_df = new_df[['interneuron', 'motor', 'sensory', 'polymodal', 'total']]

    return new_df