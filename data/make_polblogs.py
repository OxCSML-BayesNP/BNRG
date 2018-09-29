import numpy as np
import scipy.io
from parse import parse_mat
from utils.plots import *
import pickle

mat = scipy.io.loadmat('data/polblogs.mat')
graph, nodes = parse_mat(mat)
graph['labels'] = mat['Problem']['aux'][0][0][0][0][1][nodes].reshape(-1)
plt.figure('polblogs degree')
plot_degree(graph['deg'], spec='bo-')
plt.figure('polblogs communities')
plot_sorted_adj(graph, graph['labels'])
plt.show()

with open('data/polblogs.pkl', 'wb') as f:
    pickle.dump(graph, f)
