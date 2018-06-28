from model import *
from mcmc import *
from utils.ks import *
from utils.plots import *
import ignrg.mcmc
from plot_results import *
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--C', type=int, default=4)
parser.add_argument('--alpha', type=float, default=1.5)
parser.add_argument('--beta', type=float, default=3.0)
parser.add_argument('--n_chains', type=int, default=3)
parser.add_argument('--n_init', type=int, default=0)
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

argdict = vars(args)
print argdict
with open(os.path.join(savedir, 'args.txt'), 'w') as f:
    for k, v in argdict.iteritems():
        f.write(k + ':' + str(v) + '\n')

if args.data is None:
    w_true = sample_w(args.alpha, args.beta, args.n)
    s_true = exp(0.1*npr.normal())/np.sqrt(C)
    r_true = s_true*np.sqrt(C)
    V_true = sample_V(s_true, r_true, args.n, args.C)
    graph = sample_graph(w_true, V_true)
    with open(os.path.join(savedir, 'generated.pkl'), 'w') as f:
        pickle.dump(graph, f)
else:
    with open(os.path.join('../data', args.data+'.pkl'), 'r') as f:
        graph = pickle.load(f)

if args.n_init > 0:
    init_chain, w_init = ignrg.mcmc.run_mcmc(graph, args.n_init, return_w=True)
    alpha_init = init['alpha'][-1]
    beta_init = init['beta'][-1]
else:
    w_init = None
    alpha_init = None
    beta_init = None

chains = [0]*args.n_chains
for i in range(args.n_chains):
    print 'running chain %d...' % (i+1)
    chains[i] = run_mcmc(graph, args.C, args.n_samples,
            burn_in=args.burn_in, thin=args.thin,
            w=w_init, alpha=alpha_init, beta=beta_init)
    print

results = {}
results['log_joint'] = [0]*args.n_chains
results['alpha'] = []
results['beta'] = []
results['s'] = np.zeros((0, args.C))
results['r'] = np.zeros((0, args.C))
results['V_map'] = [0]*args.n_chains
results['wV_map'] = [0]*args.n_chains
for i in range(args.n_chains):
    results['alpha'] += chains[i]['alpha']
    results['beta'] += chains[i]['beta']
    results['log_joint'][i] = chains[i]['log_joint']
    results['s'] = np.r_[results['s'], chains[i]['s']]
    results['r'] = np.r_[results['r'], chains[i]['r']]
    results['V_map'][i] = chains[i]['V_map']
    results['wV_map'][i] = chains[i]['wV_map']

n_post_samples = len(results['alpha'])
degrees = np.zeros((n_post_samples, graph['n']), dtype=int)
results['ks'] = np.zeros(n_post_samples)
results['rks'] = np.zeros(n_post_samples)
print 'sampling predictive degree distributions...'
for i in range(n_post_samples):
    if (i+1)%100 == 0:
        print '%d/%d' % (i+1, n_post_samples)
    w = sample_w(results['alpha'][i], results['beta'][i], graph['n'])
    V = sample_V(results['s'][i], results['r'][i], graph['n'], args.C)
    degrees[i] = sample_graph(w, V)['deg']
    results['ks'][i] = compute_ks(graph['deg'], degrees[i])
    results['rks'][i] = compute_ks(graph['deg'], degrees[i], True)
results['observed_degree'] = graph['deg']
results['pred_degree_CI'] = compute_degree_CI(degrees)

with open(os.path.join(savedir, 'results.pkl'), 'w') as f:
    pickle.dump(results, f)

if args.plot:
    plot_results(results, graph, savedir)
else:
    print 'average KS statistic: %f (%f)' \
            % (results['ks'].mean(), results['ks'].std())
    print 'average reweighted KS statistic: %f (%f)' \
            % (results['rks'].mean(), results['rks'].std())
