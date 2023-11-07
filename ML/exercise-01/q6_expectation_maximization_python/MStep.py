import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N, D = X.shape
    K = gamma.shape[1]
    weights = np.zeros(K)
    means = np.zeros((K, D))
    covariances = np.zeros((D, D, K))

    for k in range(K):
        Nk = np.sum(gamma[:, k])
        weights[k] = Nk / N
        means[k] = np.dot(gamma[:, k], X) / Nk
        for n in range(N):
            covariances[:, :, k] += gamma[n, k] * np.dot((X[n] - means[k]).reshape(D, 1), (X[n] - means[k]).reshape(1, D))
        covariances[:, :, k] /= Nk
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
