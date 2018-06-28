import numpy as np
from utils.plots import *
import pickle
from scipy.sparse import csr_matrix

node_to_comm = {}
comm_to_nodes = {}
csizes = []
with open('com-dblp.top5000.cmty.txt', 'r') as  f:
    for i, line in enumerate(f):
        nodes = [int(l) for l in line.split()]
        nodes = nodes[:len(nodes)/5+1]
        comm_to_nodes[i] = nodes
        csizes.append(len(nodes))
        for n in nodes:
            node_to_comm[n] = i

top_comms = np.argsort(csizes)[-6:-3]
relabeled = {}
for i, k in enumerate(top_comms):
    relabeled[k] = i

raw = np.loadtxt('com-dblp.ungraph.txt', dtype=int)
filtered = []
for r, c in raw:
    k1 = node_to_comm.get(r, -1)
    k2 = node_to_comm.get(c, -1)
    if k1 in top_comms and k2 in top_comms:
        filtered.append([r, c])

filtered = np.array(filtered)
nodes = np.unique(filtered[:,[0,1]])
n = len(nodes)
labels = np.zeros(n, dtype=int)
for i in range(n):
    filtered[filtered==nodes[i]] = i
    labels[i] = relabeled[node_to_comm[nodes[i]]]

np.savetxt('dblp.txt', filtered, fmt='%d')
np.savetxt('dblp_labels.txt', labels, fmt='%d')

graph = {}
graph['n'] = n
row = filtered[:,0]
col = filtered[:,1]
graph['i'] = row
graph['j'] = col
graph['labels'] = labels
graph['deg'] = np.zeros(n, dtype=int)
for (i, j) in zip(row, col):
    graph['deg'][i] += 1
    graph['deg'][j] += 1

print 'Number of nodes: %d' % n
print 'Number of edges: %d' % len(row)

plt.figure()
plot_degree(graph['deg'])

plt.figure()
G = csr_matrix((np.ones(len(row)), (row, col)), shape=[n,n], dtype=int)
G = G + G.T
order = np.argsort(labels)
plt.spy(G[order][:,order], markersize=0.1)
plt.show()

with open('dblp.pkl', 'w') as f:
    pickle.dump(graph, f)
