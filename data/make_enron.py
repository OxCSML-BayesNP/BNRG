import numpy as np
import pickle
from parse import parse_raw
from utils.plots import *

graph, _ = parse_raw(np.loadtxt('data/enron.txt'))
plt.figure('enron degree')
plot_degree(graph['deg'], spec='bo-')
plt.show()

with open('data/enron.pkl', 'wb') as f:
    pickle.dump(graph, f)
