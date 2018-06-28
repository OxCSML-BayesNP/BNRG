from model import *
from mcmc import run_mcmc
from utils.plots import compute_degree_CI
from utils.ks import compute_ks
from plot_results import plot_results
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--alpha', type=float, default=1.5)
parser.add_argument('--beta', type=float, default=3.0)
parser.add_argument('--n_chains', type=int, default=3)
parser.add_argument('--n_samples', type=int, default=10000)
parser.add_argument('--burn_in', type=int, default=None)
parser.add_argument('--thin', type=int, default=10)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

savedir = args.savedir
if savedir is None:
    savedir = './results'
    if args.data is None:
        savedir = os.path.join(savedir, 'generated')
    else:
        savedir = os.path.join(savedir, args.data)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# save arguments
argdict = vars(args)
print argdict
with open(os.path.join(savedir, 'args.txt'), 'w') as f:
    for k, v in argdict.iteritems():
        f.write(k + ':' + str(v) + '\n')

## generate data if args.data is not given
if args.data is None:
    w_true = sample_w(args.alpha, args.beta, args.n)
    graph = sample_graph(w_true)
    with open(os.path.join(savedir, 'generated.pkl'), 'w') as f:
        pickle.dump(graph, f)
else:
    with open(os.path.join('../data', args.data+'.pkl'), 'r') as f:
        graph = pickle.load(f)

chains = [0]*args.n_chains
for i in range(args.n_chains):
    print 'running chain %d...' % (i+1)
    chains[i] = run_mcmc(graph, args.n_samples,
            burn_in=args.burn_in, thin=args.thin)
    print

results = {}
results['log_joint'] = [0]*args.n_chains
results['alpha'] = []
results['beta'] = []
for i in range(args.n_chains):
    results['alpha'] += chains[i]['alpha']
    results['beta'] += chains[i]['beta']
    results['log_joint'][i] = chains[i]['log_joint']

n_post_samples = len(results['alpha'])
degrees = np.zeros((n_post_samples, graph['n']), dtype=int)
results['ks'] = np.zeros(n_post_samples)
results['rks'] = np.zeros(n_post_samples)
print 'sampling predictive degree distributions...'
for i in range(n_post_samples):
    if (i+1)%100 == 0:
        print '%d/%d' % (i+1, n_post_samples)
    w = sample_w(results['alpha'][i], results['beta'][i], graph['n'])
    degrees[i] = sample_graph(w)['deg']
    results['ks'][i] = compute_ks(graph['deg'], degrees[i])
    results['rks'][i] = compute_ks(graph['deg'], degrees[i], True)
results['observed_degree'] = graph['deg']
results['pred_degree_CI'] = compute_degree_CI(degrees)

with open(os.path.join(savedir, 'results.pkl'), 'w') as f:
    pickle.dump(results, f)

if args.plot:
    plot_results(results, savedir)
else:
    print 'average KS statistic: %f (%f)' \
            % (results['ks'].mean(), results['ks'].std())
    print 'average reweighted KS statistic: %f (%f)' \
            % (results['rks'].mean(), results['rks'].std())
