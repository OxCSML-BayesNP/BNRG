import numpy as np
from utils.plots import *
import matplotlib.cm as cm
import os
import sys
import pickle
from matplotlib import rc, rcParams

def plot_results(results, savedir):
    rc('font', family='Dejavu Sans')
    rc('text', usetex=True)
    fig = plt.figure('log_joint')
    n_chains = len(results['log_joint'])
    colors = cm.rainbow(np.linspace(0, 1, n_chains))
    for i in range(n_chains):
        plt.plot(results['log_joint'][i], color=colors[i],
                linewidth=0.5, label='chain %d'%(i+1))
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, 'log_joint.pdf'))

    fig = plt.figure('alpha')
    plt.hist(results['alpha'], ec='m', fc='r', alpha=0.5, bins=30)
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel(r'Number of samples', fontsize=20)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, 'alpha.pdf'))

    fig = plt.figure('beta')
    plt.hist(results['beta'], ec='m', fc='r', alpha=0.5, bins=30)
    plt.xlabel(r'$\beta$', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel(r'Number of samples', fontsize=20)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, 'beta.pdf'))

    fig = plt.figure('pred_degree')
    plot_degree_CI(precomputed=results['pred_degree_CI'], label=r'pred')
    plot_degree(results['observed_degree'], spec='ro', label=r'data')
    plt.xlabel(r'Degree', fontsize=20)
    plt.ylabel(r'Distribution', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, 'pred_degree.pdf'))
    plt.show()

    print 'average KS statistic: %f (%f)' \
            % (results['ks'].mean(), results['ks'].std())
    print 'average reweighted KS statistic: %f (%f)' \
            % (results['rks'].mean(), results['rks'].std())

if __name__ == '__main__':
    assert(len(sys.argv) > 1)
    data = sys.argv[1]
    savedir = os.path.join('results', data)
    with open(os.path.join(savedir, 'results.pkl'), 'r') as f:
        results = pickle.load(f)
    plot_results(results, savedir)
