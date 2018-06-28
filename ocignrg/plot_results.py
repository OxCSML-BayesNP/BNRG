import numpy as np
from utils.plots import *
import matplotlib.cm as cm
import os
import sys
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

def plot_results(results, graph, savedir, labels=None):
    fig = plt.figure('log_joint')
    n_chains = len(results['log_joint'])
    colors = cm.rainbow(np.linspace(0, 1, n_chains))
    for i in range(n_chains):
        plt.plot(results['log_joint'][i], color=colors[i],
                linewidth=0.5, label='chain %d'%(i+1))
    plt.legend()
    fig.savefig(os.path.join(savedir, 'log_joint.pdf'))

    fig = plt.figure('alpha')
    plt.hist(results['alpha'], ec='m', fc='r', alpha=0.5, bins=30)
    fig.savefig(os.path.join(savedir, 'alpha.pdf'))

    fig = plt.figure('beta')
    plt.hist(results['beta'], ec='m', fc='r', alpha=0.5, bins=30)
    fig.savefig(os.path.join(savedir, 'beta.pdf'))

    fig = plt.figure('pred_degree')
    plot_degree_CI(precomputed=results['pred_degree_CI'])
    plot_degree(results['observed_degree'], spec='ro')
    fig.savefig(os.path.join(savedir, 'pred_degree.pdf'))

    n_chains = len(results['wV_map'])
    for i in range(n_chains):
        fig = plt.figure('wV_map_chain'+str(i+1))
        wV = results['wV_map'][i]
        ind = np.argmax(wV, 1)
        if labels is not None:
            print 'chain %d ARI: %f' % (i+1, adjusted_rand_score(ind, labels))
            #np.savetxt('../../../matlab/SNetOC/plabels%d.txt'%i, ind, fmt='%d')
        plot_sorted_adj(graph, wV)

        #order = np.argsort(ind)

        #G = csr_matrix((np.ones(len(graph['i'])), (graph['i'], graph['j'])),
        #        shape=[graph['n'], graph['n']], dtype=int)
        #G = G[order][:,order]
        #G = G + G.T
        #plt.spy(G, markersize=1.0)
        fig.savefig(os.path.join(savedir, 'sorted_adj%d.pdf' % i),
                bbox_inches='tight', pad_inches=0)
    plt.show()

    print 'average KS statistic: %f (%f)' \
            % (results['ks'].mean(), results['ks'].std())
    print 'average reweighted KS statistic: %f (%f)' \
            % (results['rks'].mean(), results['rks'].std())

if __name__ == '__main__':
    assert(len(sys.argv) > 1)
    savedir = os.path.join('./results', sys.argv[1])
    with open(os.path.join(savedir, "results.pkl"), "r") as f:
        results = pickle.load(f)
    with open(os.path.join('../data', sys.argv[1]+'.pkl'), 'r') as f:
        graph = pickle.load(f)
    labelfilename = os.path.join('../data', sys.argv[1]+'_labels.txt')
    if os.path.isfile(labelfilename):
        labels = np.loadtxt(labelfilename, dtype=int)
    else:
        labels = None

    plot_results(results, graph, savedir, labels=labels)
