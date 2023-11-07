import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    N, D = X.shape
    K = len(weights)
    gamma = np.zeros((N, K))
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    for n in range(N):
        sum = 0
        for k in range(K):
            gamma[n, k] = weights[k] * np.exp(-0.5 * np.dot(np.dot((X[n] - means[k]), np.linalg.inv(covariances[:, :, k])), (X[n] - means[k]).T)) / (np.sqrt((2 * np.pi) ** D * np.linalg.det(covariances[:, :, k])))
            sum += gamma[n, k]
        gamma[n, :] /= sum
    return [logLikelihood, gamma]
