import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created

    pos = np.arange(-5, 5.0, 0.1)
    estDensity = np.zeros((len(pos), 2))
    estDensity[:, 0] = pos
    for i in range(len(pos)):
        for j in range(len(samples)):
            estDensity[i, 1] += np.exp(-0.5 * ((pos[i] - samples[j]) / h) ** 2) / (np.sqrt(2 * np.pi) * h)
    estDensity[:, 1] /= len(samples)
    return estDensity
