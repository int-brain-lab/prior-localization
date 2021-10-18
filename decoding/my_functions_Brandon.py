import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def combine_layers_cortex(regions, delete_duplicates=False):
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    for i, region in enumerate(regions):
        for j, char in enumerate(remove):
            regions[i] = regions[i].replace(char, '')
    if delete_duplicates:
        regions = list(set(regions))
    return regions

def check_trials(trials):

    if trials is None:
        print('trials Bunch is None type')
        return False
    if trials.probabilityLeft is None:
        print('trials.probabilityLeft is None type')
        return False
    if len(trials.probabilityLeft[0].shape) > 0:
        print('trials.probabilityLeft is an array of tuples')
        return False
    if trials.probabilityLeft[0] != 0.5:
        print('trials.probabilityLeft does not start with 0.5')
        return False
    if ((not hasattr(trials, 'stimOn_times'))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        print('stimOn_times do not match with probabilityLeft')
        return False
    return True

def check_probe(probe, clusters, spikes):
    '''return True is clusters has data, histology, and position data
        depends on combine_layers_cortex for assertion check'''
    # Check if data is available for this probe
    if probe not in clusters.keys():
        print('No data available for this probe')
        return False

    # Check if histology is available for this probe
    if not hasattr(clusters[probe], 'acronym'):
        print('No histology available for this probe')
        return False
    
    # Check if metrics.cluster_id is available for this probe
    if not hasattr(clusters[probe], 'metrics'):
        print('No metrics available for this probe')
        return False
    if not hasattr(clusters[probe]['metrics'], 'cluster_id'):
        print('No cluster_ids available for this probe')
        return False

    # Check if atlas spacial positions are available for this probe
    if not (hasattr(clusters[probe], 'x') and hasattr(clusters[probe], 'y') and hasattr(clusters[probe], 'z')):
        print('No positions available for this probe')
        return False
    
    if not hasattr(spikes[probe], 'times'):
        print('No spike times available for this probe')
        return False
    
    if not hasattr(spikes[probe], 'clusters'):
        print('No spike clusters available for this probe')
        return False
    
    assert np.all(np.unique(combine_layers_cortex(clusters[probe]['acronym'])) == combine_layers_cortex(np.unique(clusters[probe]['acronym'])))
    
    return True

def getdists(xyz):
    
    xs, ys, zs = xyz[0], xyz[1], xyz[2]
    xs, ys, zs = xs - np.mean(xs), ys - np.mean(ys), zs - np.mean(zs)

    X = np.zeros((3,len(xs)))
    X[0,:] = xs
    X[1,:] = ys
    X[2,:] = zs

    pca = PCA(n_components=1)
    pca.fit(X.T)
    pc1 = pca.components_[0]
    print(pc1)
    dists = np.dot(pc1,X)
    
    return dists

def getatlassummary(xyz):
    atlascent = (np.mean(xyz[0]),np.mean(xyz[1]),np.mean(xyz[2]))
    dxyz = xyz[0] - atlascent[0],xyz[1] - atlascent[1],xyz[2] - atlascent[2]
    atlasdists = np.sqrt(dxyz[0]**2 + dxyz[1]**2 + dxyz[2]**2)
    atlasradi = np.mean(atlasdists)
    return atlascent, atlasradi

def RegPartNeurons(dists, MIN_NEURONS):
    '''returns list where each element is a group of indicies 
        referring to dists that consistitute a single partition 
        of the region'''
    nelems = len(dists)
    sinds = np.argsort(dists)
    nparts = int(nelems/MIN_NEURONS)
    if nparts == 1:
        startInd = int((nelems - MIN_NEURONS)/2)
        if np.mod(nelems - MIN_NEURONS, 2):
            startInd = int(startInd + np.random.binomial(1,.5))
        return [sinds[startInd:int(startInd+MIN_NEURONS)]]
    startInds = np.cumsum(np.insert(FairPart(nelems - MIN_NEURONS, nparts-1),0,0))
    endInds = startInds + MIN_NEURONS
    seInds = list(zip(startInds, endInds))
    return [sinds[np.arange(startInds[i],endInds[i],dtype=int)] for i in range(len(startInds))]

def FairPart(N, nparts):
    if nparts == 1:
        return np.array([N])
    
    minPart = int(N/nparts)
    a = np.append(FairPart(N-minPart,nparts-1), minPart)
    np.random.shuffle(a)
    return a

def RegPartDists(dists, MIN_DISTS=0.0003):
    '''returns list where each element is a group of indicies 
        referring to dists that consistitute a single partition 
        of the region'''
    
    # standardize units
    rdists = np.max(dists)-np.min(dists)
    sddists = (dists - np.min(dists))/rdists
    sdmd = MIN_DISTS/rdists
    
    nparts = int(1.0/sdmd)
    if nparts == 0:
        return [[i for i in range(len(dists))]], [], []
    if nparts == 1:
        lb, ub = (1-sdmd)/2, (1+sdmd)/2
        return [[np.nonzero((sddists>=lb) & (sddists<ub))[0]]]
    
    lbs = np.linspace(0,1-sdmd,nparts)
    ubs = lbs + sdmd
    ret = []
    for i in range(len(lbs)):
        lb, ub = lbs[i], ubs[i]
        if i == len(lbs)-1: # include upper bound
            ret.append(np.nonzero((sddists>=lb) & (sddists<=ub))[0])
        else:
            ret.append(np.nonzero((sddists>=lb) & (sddists<ub))[0])
    return ret, lbs*rdists + np.min(dists), ubs*rdists + np.min(dists)

def plotInsertion(xyz, cinds):
    xs, ys, zs = xyz[0], xyz[1], xyz[2]
    xs, ys, zs = xs - np.mean(xs), ys - np.mean(ys), zs - np.mean(zs)
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(xs, ys, zs, s=5, c='k')
       
    for ci in range(len(cinds)):
        ax.scatter(xs[cinds[ci]], ys[cinds[ci]], zs[cinds[ci]])

    ax.set_xlim(-.0005,.0005)
    ax.set_ylim(-.0005,.0005)
    ax.set_zlim(-.001,.001)
    plt.show()
    return