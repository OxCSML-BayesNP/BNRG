import pickle
from model import *
from utils.ks import compute_ks
import os
import sys

with open(os.path.join('../data', sys.argv[1]+'.pkl'), 'r') as f:
    graph = pickle.load(f)

filename = os.path.join('results', sys.argv[1], 'results.pkl')
with open(filename, 'r') as f:
    results = pickle.load(f)
n_post_samples = len(results['alpha'])
results['rks'] = np.zeros(n_post_samples)
for i in range(n_post_samples):
    if (i+1)%100 == 0:
        print '%d/%d' % (i+1, n_post_samples)
    w = sample_w(results['alpha'][i], results['beta'][i], graph['n'])
    degree = sample_graph(w)['deg']
    results['rks'][i] = compute_ks(graph['deg'], degree, True)
print 'Average reweighted KS: %f (%f)' % (results['rks'].mean(), results['rks'].std())

with open(filename, 'w') as f:
    pickle.dump(results, f)
