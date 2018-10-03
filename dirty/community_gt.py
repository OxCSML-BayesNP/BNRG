import numpy as np
import pickle
from utils.plots import *
import sys
import os
data = sys.argv[1]

with open(os.path.join('data', data+'.pkl'), 'rb') as f:
    graph = pickle.load(f)

n = graph['n']
deg = graph['deg']
labels = graph['labels']
ulabels = np.unique(labels)
c = len(ulabels)
U = np.zeros((n, c))
for i in range(c):
    ind = labels==ulabels[i]
    U[ind, i] = deg[ind]
plot_sorted_adj(graph, graph['labels'], U=U)
plt.savefig(os.path.join('figures', data+'_gt.pdf'),
        bbox_inches='tight', pad_inches=0)
plt.show()
