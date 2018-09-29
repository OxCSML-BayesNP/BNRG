import numpy as np
import scipy.io
import pickle
from parse import parse_mat
from utils.plots import *

graph, _ = parse_mat(scipy.io.loadmat('data/internet.mat'))
plt.figure('internet degree')
plot_degree(graph['deg'], spec='bo-')
plt.show()

with open('data/internet.pkl', 'wb') as f:
    pickle.dump(graph, f)
