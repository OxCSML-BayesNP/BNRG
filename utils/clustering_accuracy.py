import numpy as np

def clustering_accuracy(labels0, labels1):
    ulabels0 = np.unique(labels0)
    ulabels1 = np.unique(labels1)
    assert(len(ulabels0) == len(ulabels1))
    c = len(ulabels0)

    confmat = np.zeros((c, c))
    for i, u1 in enumerate(ulabels0):
        for j, u2 in enumerate(ulabels1):
            confmat[i,j] = np.logical_and(labels0==u1, labels1==u2).mean()
    return confmat.max(1).sum()

if __name__=='__main__':
    labels0 = np.loadtxt('plabels.txt')
    labels1 = np.loadtxt('labels1.txt')
    print clustering_accuracy(labels0, labels1)
