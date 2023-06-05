import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def correlation_matrix(data):
    correlation_matrix = np.corrcoef(data, rowvar=False)
    return correlation_matrix

def is_symmetric(matrix):
    transpose = np.transpose(matrix)
    return np.array_equal(matrix, transpose)

def surrogate_test(data, num_surrogates=1000):
    # Step 1: Calculate observed correlation matrix
    observed_corr = np.corrcoef(data, rowvar=False)

    # Step 2: Generate surrogate datasets
    surrogates = []
    for _ in range(num_surrogates):
        surrogate_data = data.copy()
        np.random.shuffle(surrogate_data.T) # Shuffle each column
        surrogates.append(surrogate_data)

    # Step 3: Calculate correlation matrices for surrogates
    surrogate_corrs = []
    for surrogate in surrogates:
        surrogate_corr = np.corrcoef(surrogate, rowvar=False)
        surrogate_corrs.append(surrogate_corr)

    # Step 4: Compare observed correlations to surrogates
    surrogate_corrs = np.array(surrogate_corrs)
    surrogate_summary = np.mean(surrogate_corrs, axis=0)  # Can also use np.median or other summary statistics

    # Step 5: Assess statistical significance
    corr_threshold = np.mean(surrogate_summary)

    return observed_corr, surrogate_summary, corr_threshold
