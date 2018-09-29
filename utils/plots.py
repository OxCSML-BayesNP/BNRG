import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy.stats.mstats import mquantiles
from scipy.sparse import csr_matrix

def plot_degree(x, **kwargs):
    xloglim = kwargs.get('xloglim', 16)
    step = kwargs.get('step', 1)
    spec = kwargs.get('spec', 'bo')
    label = kwargs.get('label', None)
    fontsize = kwargs.get('fontsize', 18)
    ax = kwargs.get('ax', None)

    edgebins = np.array([2**i for i in np.arange(0, xloglim+step, step)])
    sizebins = edgebins[1:] - edgebins[:-1]
    sizebins = np.append(sizebins, 1)
    centerbins = edgebins
    counts, _ = np.histogram(x, edgebins)
    counts = np.append(counts, 0)
    freq = counts.astype(float)/sizebins/len(x)
    ind = freq > 0
    centerbins = centerbins[ind]
    freq = freq[ind]

    if ax is None:
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(centerbins, freq, spec, label=label)
        if label is not None:
            plt.legend(fontsize=fontsize)
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(centerbins, freq, spec, label=label)
        if label is not None:
            ax.legend(fontsize=fontsize)

    return freq, centerbins

def compute_degree_CI(X, xloglim=16, step=1):
    edgebins = np.array([2**i for i in np.arange(0, xloglim+step, step)])
    sizebins = edgebins[1:] - edgebins[:-1]
    sizebins = np.append(sizebins, 1)
    centerbins = edgebins

    freqs = np.zeros((len(X), len(centerbins)))
    for i in range(len(X)):
        counts, _ = np.histogram(X[i], edgebins)
        counts = np.append(counts, 0)
        freqs[i] = counts.astype(float)/sizebins/len(X[i])
    qnts = mquantiles(freqs, [0.025, 0.975], axis=0, alphap=0.5, betap=0.5)
    return qnts, centerbins

def plot_degree_CI(**kwargs):
    X = kwargs.get('degrees', None)
    precomputed = kwargs.get('precomputed', None)
    xloglim = kwargs.get('xloglim', 16)
    step = kwargs.get('step', 1)
    alpha = kwargs.get('alpha', 0.3)
    label = kwargs.get('label', None)
    fontsize = kwargs.get('fontsize', 18)
    ax = kwargs.get('ax', None)

    qnts, centerbins = compute_degree_quantiles(X, xloglim=xloglim, step=step) \
            if precomputed is None else precomputed

    lb = qnts[0,:]
    ub = qnts[1,:]
    ind = ub > 0
    ub = ub[ind]
    lb = lb[ind]
    centerbins = centerbins[ind]
    ind = lb==0
    lb[ind] = 0.9*ub[ind]

    if ax is None:
        plt.xscale('log')
        plt.yscale('log')
        plt.fill_between(centerbins, lb, ub,
                alpha=alpha, label=label)
        if label is not None:
            plt.legend(fontsize=fontsize)
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.fill_between(centerbins, lb, ub,
                alpha=alpha, label=label)
        if label is not None:
            ax.legend(fontsize=fontsize)

def plot_sorted_adj(graph, labels):
    G = csr_matrix((np.ones(len(graph['i'])), (graph['i'], graph['j'])),
            shape=[graph['n'], graph['n']], dtype=int)
    G = G + G.T
    order = np.argsort(labels)
    plt.spy(G[order][:,order], markersize=1.0)

   # order = np.zeros(0, dtype=int)
   # bdrs = []
   # sz = 0
   # for k in range(c):
   #     ind = np.arange(n)[labels==k]
   #     vec = U[ind,k]
   #     suborder = np.argsort(-vec)
   #     order = np.append(order, ind[suborder])
   #     bdrs.append(sz + len(vec))
   #     sz += len(vec)

    #for bdr in bdrs:
    #    plt.axvline(x=bdr, color='k', linewidth=3.0)
    #    plt.axhline(y=bdr, color='k', linewidth=3.0)
