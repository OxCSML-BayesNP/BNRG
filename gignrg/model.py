from utils.defs import *
import scipy.sparse
from scipy.stats import poisson
from scipy.special import kv
import ctypes

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

def sample_w(nu, a, b, n):
    w = np.zeros(n)
    ctypes.CDLL("./gigrnd.so").gigrnd(
            ctypes.c_double(nu),
            ctypes.c_double(a),
            ctypes.c_double(b),
            ctypes.c_int(n),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return w

def log_prior(w, nu, a, b):
    lp = 0.5*nu*log(a/b) + (nu-1)*log(w) - 0.5*(a*w + b/w) \
            - log(2*kv(nu, np.sqrt(a*b)))
    return lp.sum()

def log_prior_grad(w, nu, a, b):
    gw = (nu-1)/w - 0.5*a + 0.5*b/(w**2)
    return gw

def log_joint_aux(graph, m, w, nu, a, b):
    return log_likel_aux(graph, m, w) + log_prior(w, nu, a, b)

def log_joint_aux_grad(graph, m, w, nu, a, b):
    return log_likel_aux_grad(graph, m, w) + log_prior_grad(w, nu, a, b)

def log_joint_aux_uc(graph, m, hw, nu, a, b):
    return log_joint_aux(graph, m, exp(hw), nu, a, b) + hw.sum()

def log_joint_aux_uc_grad(graph, m, hw, nu, a, b):
    w = exp(hw)
    gw = log_joint_aux_grad(graph, m, w, nu, a, b)
    ghw = gw*w + 1
    return ghw
