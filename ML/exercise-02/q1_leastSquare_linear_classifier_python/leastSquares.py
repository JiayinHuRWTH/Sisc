import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    weight = np.linalg.inv(data.T @ data) @ data.T @ label
    bias = weight[0]
    weight = weight[1:]
    return weight, bias
