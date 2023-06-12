import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(data1, data2):

    sim = data1.T @ data2 # all similarities without normalization
    sim = np.diagonal(sim) # trace of the similarity matrix
    norms = np.linalg.norm(data1, axis=0) * np.linalg.norm(data2, axis=0) # all norms
    norms = np.where(norms == 0, 1, norms) # avoid division by zero

    return sim/norms