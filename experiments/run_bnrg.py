from model.bnrg import *
from model.transformed_ig import *
from model.transformed_gig import *
from mcmc.bnrg_mcmc import run_mcmc
from utils.plots import *
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--prior', type=str, default='IG')
parser.add_argument('--n_chains', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=5000)
parser.add_argument('--thin', type=int, default=10)
parser.add_argument('--exp_name', type=str, default='trial')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

if args.data is None:
    raise ValueError('You must specify data')
with open(os.path.join('./data', args.data+'.pkl'), 'rb') as f:
    graph = pickle.load(f)
save_dir = os.path.join('./results/bnrg', args.prior, args.data, args.exp_name)

if args.prior == 'IG':
    phw = TransformedIG(graph=graph)
elif args.prior == 'GIG':
    phw = TransformedGIG(graph=graph)
else:
    raise ValueError('Invalid prior {}'.format(args.prior))
model = BNRG(phw)

def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logfile = open(os.path.join(save_dir, 'train.log'), 'w', 0)
    chains = []
    for i in range(1, args.n_chains+1):
        print 'running chain {}'.format(i)
        chains.append(run_mcmc(graph, model,
            args.n_samples, thin=args.thin, logfile=logfile))

    results = {}
    for i in range(args.n_chains):
        for key in chains[i].keys():
            if results.get(key) is None:
                results[key] = [chains[i][key]]
            else:
                results[key].append(chains[i][key])
    results['pred_degree_CI'] = compute_degree_CI(
            np.concatenate(results['pred_degree']))
    results.pop('pred_degree')
    results['observed_degree'] = graph['deg']
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def show():
    with open(os.path.join(save_dir, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    print 'average reweighted KS statistics: {:.4f} ({:.4f})'.format(
            np.mean(results['rks']), np.std(results['rks']))
    if args.plot:
        fig = plt.figure('pred_degree')
        plot_degree_CI(precomputed=results['pred_degree_CI'], label=r'pred')
        plot_degree(results['observed_degree'], spec='ro', label=r'data')
        plt.xlabel(r'Degree', fontsize=20)
        plt.ylabel(r'Distribution', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'show':
        show()
