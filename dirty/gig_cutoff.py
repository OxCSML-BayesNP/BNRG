import numpy as np
from model.transformed_gig import TransformedGIG
from model.bnrg import BNRG
from matplotlib import rc, rcParams
from utils.plots import *
from scipy.special import kv
log = np.log
exp = np.exp
from matplotlib import rc, rcParams

rc('font', family='Dejavu Sans')
rc('text', usetex=True)

# get theoretically expected asymptotic degree distribution
def plot_theory_degree(nu, a, b, **kwargs):
    xloglim = kwargs.get('xloglim', 10)
    step = kwargs.get('step', 1)
    spec = kwargs.get('spec', 'bo')
    linewidth = kwargs.get('linewidth', 0.5)
    label = kwargs.get('label', None)
    fontsize = kwargs.get('fontsize', 18)
    ax = kwargs.get('ax', None)

    x = np.array([2**i for i in np.arange(0, xloglim+step, step)])
    log_y = 0.5*nu*(log(a)-log(b)) \
            - log(2*kv(nu, np.sqrt(a*b))) - nu*log(1 + 0.5*a) \
            + (nu-1)*log(x) - log(1 + 0.5*a)*x
    y = exp(log_y)
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

nu = -0.5
a = 0.01
b = 1.0
n_list = [int(1e+3), int(1e+4), int(1e+5), int(1e+6)]
spec_list = ['bo', 'rs', 'gv', 'md']

plt.figure('GIG')
for n, spec in zip(n_list, spec_list):
    _, w = TransformedGIG.sample_(nu, a, b, n)
    deg = BNRG.sample_graph(w)['deg']
    plot_degree(deg, spec=spec, step=1.0,
            label=r'$n={}$'.format(n), fontsize=18)

plot_theory_degree(nu, 1e-5, b, spec='k--', xloglim=14, linewidth=1.0)
plt.xlabel(r'Degree', fontsize=18)
plt.ylabel(r'Distribution', fontsize=18)
plt.savefig('figures/gig_cutoff.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
