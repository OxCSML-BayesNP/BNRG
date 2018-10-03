import pickle
import os
import sys
from utils.plots import *
from matplotlib import rc, rcParams

rc('font', family='Dejavu Sans')
rc('text', usetex=True)

data = sys.argv[1]
with open(os.path.join('data', data+'.pkl'), 'rb') as f:
    graph = pickle.load(f)
plot_degree(graph['deg'], spec='ro', label=r'data')
plt.xlabel(r'Degree', fontsize=16)
plt.ylabel(r'Distribution', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

with open(os.path.join('results/cbnrg/IG',
    data, 'trial', 'results.pkl'), 'rb') as f:
    results = pickle.load(f)
plot_degree_CI(precomputed=results['pred_degree_CI'], label=r'IG')

with open(os.path.join('results/cbnrg/GIG',
    data, 'trial', 'results.pkl'), 'rb') as f:
    results = pickle.load(f)
plot_degree_CI(precomputed=results['pred_degree_CI'], label=r'GIG')

matlab_root = '/Users/juho/codes/matlab/SNetOC/results'
cbins = np.loadtxt(os.path.join(matlab_root, data, 'cbins.txt'))
qnts = np.loadtxt(os.path.join(matlab_root, data, 'qnts.txt'))
plot_degree_CI(precomputed=(qnts, cbins), label=r'CGGP')

matlab_root = '/Users/juho/codes/matlab/SNetOC/MMSB_results'
cbins = np.loadtxt(os.path.join(matlab_root, data, 'cbins.txt'))
qnts = np.loadtxt(os.path.join(matlab_root, data, 'qnts.txt'))
plot_degree_CI(precomputed=(qnts, cbins), label=r'MMSB')
plt.legend(loc=3, fontsize=18)

plt.tight_layout()
plt.savefig(os.path.join('figures', data+'_degrees.pdf'),
        dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()
