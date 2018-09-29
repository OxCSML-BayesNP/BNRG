import numpy as np
import pickle
from parse import parse_raw
from utils.plots import *

graph, _ = parse_raw(np.loadtxt('data/cond_mat.txt'))
plt.figure('cond-mat degree')
plot_degree(graph['deg'], spec='bo-')
plt.show()

with open('data/cond_mat.pkl', 'wb') as f:
    pickle.dump(graph, f)
