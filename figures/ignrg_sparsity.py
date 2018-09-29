from utils.plots import *
from model.transformed_ig import TransformedIG
from model.bnrg import BNRG
from matplotlib import rc

ns = np.arange(1000, 10001, 1000)
alphas = [1.4, 2.0, 2.6]
betas = [1.0, 2.0]
emp_specs = ['bo-', 'rs-', 'gv-']
thr_specs = ['b--', 'r--', 'g--']
rc('font', family='Dejavu Sans')
rc('text', usetex=True)

fig, axarr = plt.subplots(1, 2)
for i, beta in enumerate(betas):
    axarr[i].set_title(r'$\beta=$%.1f' % beta, fontsize=25)
    for j, alpha in enumerate(alphas):
        print 'processing alpha %f, beta %f...' % (alpha, beta)
        E = []
        EE = []
        for n in ns:
            _, w = TransformedIG.sample_(alpha, beta, n)
            graph = BNRG.sample_graph(w)
            E.append(graph['n_edges'])
            EE.append(0.5*n*beta/(alpha-1))
        label = r'$\alpha=$%.1f' % alpha
        axarr[i].plot(ns, E, emp_specs[j], label=label)
        axarr[i].plot(ns, EE, thr_specs[j])
        axarr[i].set_xlabel(r'Number of nodes', fontsize=18)
        axarr[i].xaxis.set_tick_params(labelsize=18)
        axarr[i].set_ylabel(r'Number of edges', fontsize=18)
        axarr[i].yaxis.set_tick_params(labelsize=18)
    x0, x1 = axarr[i].get_xlim()
    y0, y1 = axarr[i].get_ylim()
    axarr[i].set_aspect((x1-x0)/(y1-y0))
    axarr[i].legend(fontsize=18)
plt.show()
fig.savefig('figures/ignrg_sparsity.pdf', dpi=500, bbox_inches='tight', pad_inches=0)
