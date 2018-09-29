from model.bnrg import BNRG
from model.cbnrg import CBNRG
from model.transformed_ig import TransformedIG
from model.transformed_gig import TransformedGIG
from model.transformed_dir import TransformedDir
from utils.plots import *
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--c', type=int, default=None)
parser.add_argument('--gam0', type=float, default=0.1)
parser.add_argument('--prior', type=str, default='IG')
parser.add_argument('--alpha', type=float, default=1.5)
parser.add_argument('--beta', type=float, default=3.0)
parser.add_argument('--nu', type=float, default=-1.0)
parser.add_argument('--a', type=float, default=1e-2)
parser.add_argument('--b', type=float, default=2.0)
args = parser.parse_args()

if args.name is None:
    name = './data/synth_{}_{}'.format(args.prior, args.n)
    if args.prior == 'IG':
        name += '_{}_{}'.format(args.alpha, args.beta)
    elif args.prior == 'GIG':
        name += '_{}_{}_{}'.format(args.nu, args.a, args.b)
    else:
        raise ValueError('Invalid prior {}'.format(args.prior))
    if args.c is not None:
        name += '_{}_{}'.format(args.c, args.gam0)
    name += '.pkl'
else:
    name = args.name

if args.prior == 'IG':
    _, w_true = TransformedIG.sample_(args.alpha, args.beta, args.n)
elif args.prior == 'GIG':
    _, w_true = TransformedGIG.sample_(args.nu, args.a, args.b, args.n)
else:
    raise ValueError('Invalid prior {}'.format(args.prior))

if args.c is None:
    graph = BNRG.sample_graph(w_true)
else:
    _, V_true = TransformedDir.sample_(args.gam0*np.ones(args.c), args.n)
    graph = CBNRG.sample_graph(w_true, V_true)

with open(os.path.join(name), 'wb') as f:
    pickle.dump(graph, f)
