import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def correlation_matrix(data):
    correlation_matrix = np.corrcoef(data, rowvar=False)
    return correlation_matrix

def is_symmetric(matrix):
    transpose = np.transpose(matrix)
    return np.array_equal(matrix, transpose)