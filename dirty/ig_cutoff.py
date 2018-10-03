import numpy as np
from model.transformed_ig import TransformedIG
from model.bnrg import BNRG
from matplotlib import rc, rcParams
from utils.plots import *
from scipy.special import kv
log = np.log
exp = np.exp
from matplotlib import rc, rcParams
from model.defs import *

rc('font', family='Dejavu Sans')
rc('text', usetex=True)

# get theoretically expected asymptotic degree distribution
def plot_theory_degree(alpha, beta, **kwargs):
    xloglim = kwargs.get('xloglim', 8)
    step = kwargs.get('step', 1)
    spec = kwargs.get('spec', 'bo')
    label = kwargs.get('label', None)
    fontsize = kwargs.get('fontsize', 18)
    linewidth = kwargs.get('linewidth', 0.5)
    ax = kwargs.get('ax', None)

    x = np.array([2**i for i in np.arange(step, xloglim+step, step)])
    y = exp(alpha*log(beta) - (alpha+1)*log(x) - gammaln(alpha))
    if ax is None:
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(x, y, spec, label=label, linewidth=linewidth)
        if label is not None:
            plt.legend(fontsize=fontsize)
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(x, y, spec, label=label, linewidth=linewidth)
        if label is not None:
            ax.legend(fontsize=fontsize)

alpha = 1.5
beta = 1.5
n_list = [int(1e+3), int(1e+4), int(1e+5), int(1e+6)]
spec_list = ['bo', 'rs', 'gv', 'md']

plt.figure('IG')
for n, spec in zip(n_list, spec_list):
    _, w = TransformedIG.sample_(alpha, beta, n)
    deg = BNRG.sample_graph(w)['deg']
    plot_degree(deg, spec=spec, step=1.0,
            label=r'$n={}$'.format(n), fontsize=18)

plot_theory_degree(alpha, beta, spec='k--', xloglim=14, linewidth=1.0)
plt.xlabel(r'Degree', fontsize=18)
plt.ylabel(r'Distribution', fontsize=18)
plt.savefig('figures/ig_cutoff.pdf', bbox_inches='tight', pad_inches=0)
plt.show()