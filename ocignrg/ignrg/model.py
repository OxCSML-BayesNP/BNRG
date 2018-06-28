from utils.defs import *
import scipy.sparse
from scipy.stats import poisson

#def sample_graph_(w):
#    n = len(w)
#    P = 1 - exp(-np.outer(w, w)/w.sum())
#    X = (npr.rand(n, n) < P).astype(int)
#    X = np.triu(X, k=1)
#    graph = {}
#    graph['n'] = n
#    graph['i'], graph['j'], _ = scipy.sparse.find(X)
#    graph['deg'] = np.zeros(n, dtype=int)
#    for (i, j) in zip(graph['i'], graph['j']):
#        graph['deg'][i] += 1
#        graph['deg'][j] += 1
#    return graph

def sample_graph(w):
    n = len(w)
    wb = w.sum()
    E = npr.poisson(wb)
    pairs = npr.choice(n, size=[E, 2], p=w/wb)
    hashed_pairs = {}
    for (i, j) in pairs:
        if i < j and hashed_pairs.get((i, j)) is None:
            hashed_pairs[(i, j)] = 1

    graph = {}
    graph['n'] = n
    graph['i'] = np.zeros(len(hashed_pairs), dtype=int)
    graph['j'] = np.zeros(len(hashed_pairs), dtype=int)
    graph['deg'] = np.zeros(n, dtype=int)

    for k, (i, j) in enumerate(hashed_pairs):
        graph['i'][k] = i
        graph['j'][k] = j
        graph['deg'][i] += 1
        graph['deg'][j] += 1

    return graph

def log_likel(graph, w):
    wb = w.sum()
    lam = np.outer(w, w)/wb
    ll = -np.triu(lam, k=1).sum()
    lam_pos = w[graph['i']]*w[graph['j']]/wb
    ll += lam_pos.sum() + log(1 - exp(-lam_pos)).sum()
    return ll

def sample_m(graph, w):
    lam = w[graph['i']]*w[graph['j']]/w.sum()
    m = np.ones(len(lam))
    ind = lam > 1e-5
    if np.any(ind):
        n_ = sum(ind)
        lam_ = lam[ind]
        m[ind] = poisson.ppf(exp(-lam_) + npr.rand(n_)*(1-exp(-lam_)), lam_)
    return m

def log_likel_aux(graph, m, w):
    n = len(w)
    wb = w.sum()
    ll = 0.5*((w**2).sum()/wb - wb)
    wi = w[graph['i']]
    wj = w[graph['j']]
    ll += (m*(log(wi) + log(wj) - log(wb)) - gammaln(m+1)).sum()
    return ll

def log_likel_aux_grad(graph, m, w):
    n = len(w)
    wb = w.sum()
    gw = w/wb - 0.5*(w**2/wb**2).sum() - 0.5
    gw += np.bincount(graph['i'], weights=m/(w[graph['i']]+tol), minlength=n)
    gw += np.bincount(graph['j'], weights=m/(w[graph['j']]+tol), minlength=n)
    gw -= m.sum()/(wb + tol)
    return gw

def sample_w(alpha, beta, n):
    return 1./npr.gamma(alpha, 1./beta, size=n)

def log_prior(w, alpha, beta):
    lp = alpha*log(beta) - (alpha+1)*log(w) - beta/w - gammaln(alpha)
    return lp.sum()

def log_prior_grad(w, alpha, beta):
    gw = -(alpha+1)/(w + tol) + beta/w**2
    return gw

def log_joint_aux(graph, m, w, alpha, beta):
    return log_likel_aux(graph, m, w) + log_prior(w, alpha, beta)

def log_joint_aux_grad(graph, m, w, alpha, beta):
    return log_likel_aux_grad(graph, m, w) + log_prior_grad(w, alpha, beta)

def log_joint_aux_uc(graph, m, hw, alpha, beta):
    return log_joint_aux(graph, m, exp(hw), alpha, beta) + hw.sum()

def log_joint_aux_uc_grad(graph, m, hw, alpha, beta):
    w = exp(hw)
    gw = log_joint_aux_grad(graph, m, w, alpha, beta)
    ghw = gw*w + 1
    return ghw
