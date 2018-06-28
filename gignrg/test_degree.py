from utils.plots import *
from model import *
from matplotlib import rc, rcParams
from scipy.special import kv

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
    log_y = 0.5*nu*(log(a)-log(b)) - 0.5*b/x \
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

n = 10000
nu_list = [-0.5, -1.0]
a_list = [1e-2, 1e-1, 1.0]
b = 2.0
emp_specs = ['bo-', 'rs-', 'gv-']
thr_specs = ['b--', 'r--', 'g--']
rc('font', family='Dejavu Sans')
rc('text', usetex=True)
fig, axarr = plt.subplots(1, 2)
for i, nu in enumerate(nu_list):
    axarr[i].set_title(r'$\nu=$%.1f' % nu, fontsize=25)
    for j, a in enumerate(a_list):
        print 'processing nu %f, a %f...' % (nu, a)
        w = sample_w(nu, a, b, n)
        deg = sample_graph(w)['deg']
        label = r'$a=$%.2f' % a
        _, x = plot_degree(deg, label=label, spec=emp_specs[j], ax=axarr[i], step=0.5)
        plot_theory_degree(nu, a, b, spec=thr_specs[j], ax=axarr[i],
            xloglim=log(x[-1])/log(2))
        axarr[i].set_xlabel(r'Degree', fontsize=18)
        axarr[i].xaxis.set_tick_params(labelsize=18)
        axarr[i].set_ylabel(r'Distribution', fontsize=18)
        axarr[i].yaxis.set_tick_params(labelsize=18)
    x0, x1 = axarr[i].get_xlim()
    y0, y1 = axarr[i].get_ylim()
    axarr[i].set_aspect((log(x1)-log(x0))/(log(y1)-log(y0)))
plt.show()
fig.savefig('gignrg_degree.pdf', dpi=500, bbox_inches='tight', pad_inches=0)
