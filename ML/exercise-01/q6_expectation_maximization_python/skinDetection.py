import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    # compute GMM
    nweights, nmeans, ncovariances = estGaussMixEM(ndata, K, n_iter, epsilon)
    sweights, smeans, scovariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    print(nweights.shape, nmeans.shape, ncovariances.shape)
    result = np.zeros(img.shape[:2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img[i][j]
            nlogLikelihood = getLogLikelihood(nmeans, nweights, ncovariances, img[i, j])
            slogLikelihood = getLogLikelihood(smeans, sweights, scovariances, img[i, j])
            # print(nlogLikelihood, slogLikelihood)
            if np.exp(slogLikelihood) > theta * np.exp(nlogLikelihood):
                result[i, j] = 1
    return result
