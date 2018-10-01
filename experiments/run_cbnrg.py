from model.bnrg import BNRG
from model.cbnrg import CBNRG
from model.transformed_ig import *
from model.transformed_gig import *
from model.transformed_dir import *
import mcmc.cbnrg_mcmc as cbnrg_mcmc
import mcmc.bnrg_mcmc as bnrg_mcmc
from utils.plots import *
from utils.clustering_accuracy import clustering_accuracy
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--c', type=int, default=4)
parser.add_argument('--prior', type=str, default='IG')
parser.add_argument('--n_chains', type=int, default=1)
#parser.add_argument('--n_init_samples', type=int, default=0)
parser.add_argument('--burn_in', type=int, default=None)
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
save_dir = os.path.join('./results/cbnrg', args.prior, args.data, args.exp_name)

if args.prior == 'IG':
    phw = TransformedIG(graph=graph)
elif args.prior == 'GIG':
    phw = TransformedGIG(graph=graph)
else:
    raise ValueError('Invalid prior {}'.format(args.prior))
init_model = BNRG(phw)
phV = TransformedDir(args.c)
model = CBNRG(phw, phV)

def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logfile = open(os.path.join(save_dir, 'train.log'), 'w', 0)

    chains = []
    for i in range(1, args.n_chains+1):

        #if args.n_init_samples > 0:
        #    # initialize by running BNRG
        #    line = 'initializing by running BNRG for chain {}'.format(i)
        #    print line
        #    logfile.write(line+'\n')
        #    init_chain = bnrg_mcmc.run_mcmc(graph, init_model,
        #            args.n_init_samples, thin=args.thin, logfile=logfile)
        #    w_init = init_chain['w_est']
        #    model.phw.init_from_chain(init_chain)
        #else:
        #    w_init = None

        line = 'running chain {}'.format(i)
        print line
        logfile.write(line+'\n')
        chains.append(cbnrg_mcmc.run_mcmc(graph, model,
            args.n_samples, args.burn_in, thin=args.thin,
            logfile=logfile))
        print
        logfile.write('\n')

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
    i = np.argmax(results['max_lj'])
    w = results['w_est'][i]
    V = results['V_est'][i]
    results['labels'] = cbnrg_mcmc.compute_labels(w, V)

    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def show(plot=True):
    with open(os.path.join(save_dir, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    print 'average reweighted KS statistics: {:.4f} ({:.4f})'.format(
            np.mean(results['rks']), np.std(results['rks']))
    print 'clustering accuracy {:.4f}'.format(
            clustering_accuracy(results['labels'], graph['labels']))

    if plot:
        fig = plt.figure('pred_degree')
        plot_degree_CI(precomputed=results['pred_degree_CI'], label=r'pred')
        plot_degree(results['observed_degree'], spec='ro', label=r'data')
        plt.xlabel(r'Degree', fontsize=20)
        plt.ylabel(r'Distribution', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=20)
        plt.tight_layout()

        fig = plt.figure('sorted_adj')
        plot_sorted_adj(graph, results['labels'])
        plt.show()

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'show':
        show(args.plot)
