import numpy as np
import scipy.sparse

def parse_raw(raw):
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
    graph['n_edges'] = len(row)
    graph['i'] = row
    graph['j'] = col
    graph['deg'] = np.zeros(n, dtype=int)
    for (i, j) in zip(row, col):
        graph['deg'][i] += 1
        graph['deg'][j] += 1
    return graph, nodes

def parse_mat(mat):
    G = mat['Problem']['A'][0][0]
    G = G + G.T
    G = G - scipy.sparse.diags(G.diagonal())

    # extract adjacency matrix
    row, col, _ = scipy.sparse.find(G)
    raw = np.c_[row, col]
    return parse_raw(raw)
