from utils.plots import *
from model import *
from matplotlib import rc
from scipy.special import kv

ns = np.arange(1000, 10001, 1000)
nu_list = [-0.5, -1.0]
a_list = [1e-2, 1e-1, 1.0]
b = 2.0
emp_specs = ['bo-', 'rs-', 'gv-']
thr_specs = ['b--', 'r--', 'g--']
rc('font', family='Dejavu Sans')
rc('text', usetex=True)

fig, axarr = plt.subplots(1, 2)
for i,nu in enumerate(nu_list):
    axarr[i].set_title(r'$\nu=$%.1f' % nu, fontsize=25)
    for j, a in enumerate(a_list):
        print 'processing nu %f, a %f...' % (nu, a)
        E = []
        EE = []
        for n in ns:
            w = sample_w(nu, a, b, n)
            graph = sample_graph(w)
            E.append(len(graph['i']))
            EE.append(0.5*n*np.sqrt(b)*kv(nu+1, np.sqrt(a*b)) \
                    /(np.sqrt(a)*kv(nu, np.sqrt(a*b))))
        label = r'$a=$%.2f' % a
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
fig.savefig('gignrg_sparsity.pdf', dpi=500, bbox_inches='tight', pad_inches=0)
