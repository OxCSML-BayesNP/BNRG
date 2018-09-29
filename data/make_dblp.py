import numpy as np
from parse import parse_raw
from utils.plots import *
import pickle

node_to_comm = {}
comm_to_nodes = {}
csizes = []
with open('data/com-dblp.top5000.cmty.txt', 'r') as  f:
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

raw = np.loadtxt('data/com-dblp.ungraph.txt', dtype=int)
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

graph, _ = parse_raw(filtered)
graph['labels'] = labels

plt.figure('dblp degree')
plot_degree(graph['deg'], spec='bo-')
plt.figure('dblp communities')
plot_sorted_adj(graph, graph['labels'])
plt.show()

with open('data/dblp.pkl', 'wb') as f:
    pickle.dump(graph, f)
