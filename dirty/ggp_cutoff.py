import numpy as np
import os
from matplotlib import rc, rcParams
from utils.plots import *
log = np.log
exp = np.exp
from matplotlib import rc, rcParams
from scipy.special import kv

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

matlabroot = '/Users/juho/codes/matlab/bnpgraph'
deg0 = np.loadtxt(os.path.join(matlabroot, 'deg0.txt'), dtype=int)
deg1 = np.loadtxt(os.path.join(matlabroot, 'deg1.txt'), dtype=int)
deg2 = np.loadtxt(os.path.join(matlabroot, 'deg2.txt'), dtype=int)
deg3 = np.loadtxt(os.path.join(matlabroot, 'deg3.txt'), dtype=int)

plt.figure('GGP')
plot_degree(deg0, spec='bo',
        label=r'$n={}$'.format(len(deg0)), fontsize=18, xloglim=14)
plot_degree(deg1, spec='rs',
        label=r'$n={}$'.format(len(deg1)), fontsize=18, xloglim=14)
plot_degree(deg2, spec='gd',
        label=r'$n={}$'.format(len(deg2)), fontsize=18, xloglim=14)
plot_degree(deg3, spec='mv',
        label=r'$n={}$'.format(len(deg3)), fontsize=18, xloglim=14)

plot_theory_degree(-0.3, 1e-5, 1., spec='k--', xloglim=14, linewidth=1)
plt.xlabel(r'Degree', fontsize=18)
plt.ylabel(r'Distribution', fontsize=18)
plt.savefig('figures/ggp_cutoff.pdf', bbox_inches='tight', pad_inches=0)
plt.legend(loc=3, fontsize=18)
plt.show()
