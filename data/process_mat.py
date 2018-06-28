import scipy.io
import scipy.sparse
import numpy as np
import os
import sys

assert(len(sys.argv) > 1)
filename = sys.argv[1]
outfilename = os.path.splitext(filename)[0] + ".txt" \
        if len(sys.argv)==2 else sys.argv[2]

# load mat file
mat = scipy.io.loadmat(filename)
G = mat['Problem']['A'][0][0]
G = G + G.T
G = G - scipy.sparse.diags(G.diagonal())

# extract adjacency matrix
row, col, _ = scipy.sparse.find(G)
raw = np.c_[row, col]
np.savetxt(outfilename, raw, fmt='%d')
