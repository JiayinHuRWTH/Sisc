import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    if (len(X.shape) > 1):
        N, D = X.shape
    else:
        D = len(X)
        N = 1
    K = len(weights)

    logLikelihood = 0
    if (N == 1):
        for k in range(K):
            logLikelihood += weights[k] * np.exp(-0.5 * np.dot(np.dot((X - means[k]), np.linalg.inv(covariances[:, :, k])), (X - means[k]).T)) / (np.sqrt((2 * np.pi) ** D * np.linalg.det(covariances[:, :, k])))
        logLikelihood = np.log(logLikelihood)
    else:
        for n in range(N):
            temp = 0
            for k in range(K):
                temp += weights[k] * np.exp(-0.5 * np.dot(np.dot((X[n] - means[k]), np.linalg.inv(covariances[:, :, k])), (X[n] - means[k]).T)) / (np.sqrt((2 * np.pi) ** D * np.linalg.det(covariances[:, :, k])))
            logLikelihood += np.log(temp)
    return logLikelihood

