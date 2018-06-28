import numpy as np
import numpy.random as npr

def catrnd(P):
    s = P.cumsum(1)
    r = npr.rand(P.shape[0], 1)
    x = (s < r).sum(1)
    return np.eye(P.shape[1])[x]

def mnrnd(ntrials, P):
    ind = np.repeat(np.arange(len(ntrials)), ntrials)
    c = catrnd(P[ind]).astype(bool)
    x = np.zeros((len(ntrials), P.shape[1]))
    for i in range(P.shape[1]):
        x[:,i] = np.bincount(ind[c[:,i]], minlength=len(ntrials))
    return x

if __name__ == '__main__':
    P = np.array([[0.2, 0.3, 0.5],
        [0.4, 0.5, 0.1],
        [0.2, 0.4, 0.4]])
    ntrials = np.array([10000, 1000000, 10000])
    x = mnrnd(ntrials, P)
    print x/ntrials[:,None]
