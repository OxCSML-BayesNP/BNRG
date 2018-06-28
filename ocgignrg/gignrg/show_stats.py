import numpy as np
import os
import sys
import pickle
from scipy.stats.mstats import mquantiles

data = sys.argv[1]
savedir = os.path.join('results', data)
with open(os.path.join(savedir, 'results.pkl'), 'r') as f:
    results = pickle.load(f)

qnts = mquantiles(results['nu'], [0.025, 0.975], alphap=0.5, betap=0.5)
print 'nu: %f, %f' % (qnts[0], qnts[1])

qnts = mquantiles(results['a'], [0.025, 0.975], alphap=0.5, betap=0.5)
print 'a: %f, %f' % (qnts[0], qnts[1])

qnts = mquantiles(results['b'], [0.025, 0.975], alphap=0.5, betap=0.5)
print 'b: %f, %f' % (qnts[0], qnts[1])

qnts = mquantiles(results['ks'], [0.025, 0.975], alphap=0.5, betap=0.5)
print 'KS: %f, %f' % (qnts[0], qnts[1])
print 'average KS statistic: %f (%f)' \
        % (results['ks'].mean(), results['ks'].std())
print 'average reweighted KS statistic: %f (%f)' \
        % (results['rks'].mean(), results['rks'].std())
