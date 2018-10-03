from utils.plots import *
from model.defs import *
from model.transformed_ig import TransformedIG
from model.transformed_dir import TransformedDir
from model.cbnrg import CBNRG
from matplotlib import rc, rcParams

def plot_theory_degree(alpha, beta, **kwargs):
    xloglim = kwargs.get('xloglim', 8)
    step = kwargs.get('step', 1)
    spec = kwargs.get('spec', 'bo')
    label = kwargs.get('label', None)
    fontsize = kwargs.get('fontsize', 18)
    linewidth = kwargs.get('linewidth', 0.5)
    ax = kwargs.get('ax', None)

    x = np.array([2**i for i in np.arange(step, xloglim+step, step)])
    y = exp(alpha*log(beta) - (alpha+1)*log(x) - beta/log(x) - gammaln(alpha))
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

n = 50000
c = 5
gam = 0.1*np.ones(c)
alphas = [1.4, 2.0, 2.6]
betas = [1.0, 2.0]
emp_specs = ['bo', 'rs', 'gv']
thr_specs = ['b--', 'r--', 'g--']
rc('font', family='Dejavu Sans')
rc('text', usetex=True)
fig, axarr = plt.subplots(1, 2)
for i, beta in enumerate(betas):
    axarr[i].set_title(r'$\beta=$%.1f' % beta, fontsize=25)
    for j, alpha in enumerate(alphas):
        print 'processing alpha %f, beta %f...' % (alpha, beta)
        _, w = TransformedIG.sample_(alpha, beta, n)
        _, V = TransformedDir.sample_(gam, n)
        graph = CBNRG.sample_graph(w, V)
        deg = graph['deg']
        label = r'$\alpha=$%.1f' % alpha
        _, x = plot_degree(deg, label=label, spec=emp_specs[j], ax=axarr[i])
        plot_theory_degree(alpha, beta, spec=thr_specs[j], ax=axarr[i],
                xloglim=log(x[-1])/log(2))
        axarr[i].set_xlabel(r'Degree', fontsize=18)
        axarr[i].xaxis.set_tick_params(labelsize=18)
        axarr[i].set_ylabel(r'Distribution', fontsize=18)
        axarr[i].yaxis.set_tick_params(labelsize=18)
    x0, x1 = axarr[i].get_xlim()
    y0, y1 = axarr[i].get_ylim()
    axarr[i].set_aspect((log(x1)-log(x0))/(log(y1)-log(y0)))
plt.show()
fig.savefig('figures/igcnr_degree.pdf', dpi=500, bbox_inches='tight', pad_inches=0)
