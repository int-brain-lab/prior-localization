import numpy as np

def __fakeneuron__(win=60, l=800, sml=20, meanfrate=10):
    sml = sml + 1 + (sml%2) # raise to next odd number
    n_sml = int(sml/2)

    binEdges = np.nonzero(np.random.poisson(lam=1.0/win, size=(l + (2*n_sml))))[0] - n_sml
    binEdges = np.insert(binEdges, 0, 0)
    binEdges = np.append(binEdges, l+(2*n_sml)-1)
#     print(binEdges)

    binVals = meanfrate*2*np.random.rand(len(binEdges)-1)
    
    fakeData = np.zeros(l + (2*n_sml))
    for j in range(len(binVals)):
        leftE, rightE = binEdges[j], binEdges[j+1]
        fakeData[leftE:rightE] = binVals[j]

    fakeData = np.random.poisson(np.convolve(fakeData, np.ones(sml)/sml, mode='valid'))
#     print(len(fakeData))
    return fakeData

def data_uncorr(N_NEURONS,T, mfr):
    X = np.zeros((N_NEURONS,T))
    for i in range(N_NEURONS):
        X[i,:] = __fakeneuron__(l=T, meanfrate=mfr)
    return X

def data_corr(y, N_NEURONS,T, mfr, c=0.1):
    '''
    generate data correlated with y where y is an array of positive values
    '''
    assert len(y)==T
    X = np.zeros((N_NEURONS,T))
    for i in range(N_NEURONS):
        X[i,:] = __fakeneuron__(l=T, meanfrate=mfr) + c*np.random.exponential(y)
    return X