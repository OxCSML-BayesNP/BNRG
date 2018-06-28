import pickle
import numpy as np
import os
import sys
from utils.plots import *

assert(len(sys.argv) > 1)
filename = sys.argv[1]
outfilename = os.path.splitext(filename)[0] + ".pkl" \
        if len(sys.argv)==2 else sys.argv[2]

# get upper triangular part
raw = np.loadtxt(filename)
row = raw[:,0]
col = raw[:,1]
ind = row < col
row = row[ind].astype(int)
col = col[ind].astype(int)

# remove nodes with no connection
nodes = np.unique(raw[:,[0,1]])
n = len(nodes)
for i in range(n):
    row[row==nodes[i]] = i
    col[col==nodes[i]] = i

graph = {}
graph['n'] = n
graph['i'] = row
graph['j'] = col
graph['deg'] = np.zeros(n, dtype=int)
for (i, j) in zip(row, col):
    graph['deg'][i] += 1
    graph['deg'][j] += 1

print 'Filename: %s' % filename
print 'Output filename: %s' % outfilename
print 'Number of nodes: %d' % n
print 'Number of edges: %d' % len(row)

plot_degree(graph['deg'])
plt.show()

with open(outfilename, "w") as f:
    pickle.dump(graph, f)
