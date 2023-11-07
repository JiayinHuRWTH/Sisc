import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    pos = np.arange(-5, 5.0, 0.1)
    distances = np.zeros((len(pos), len(samples)))
    estDensity = np.zeros((len(pos), 2))
    estDensity[:, 0] = pos
    for i in range(len(pos)):
        for j in range(len(samples)):
            distances[i, j] = np.abs(pos[i] - samples[j])
    for i in range(len(pos)):
        distances[i, :] = np.sort(distances[i, :])
        estDensity[i, 1] = k / (2 * distances[i, k - 1])
    estDensity[:, 1] /= len(samples)
    return estDensity
