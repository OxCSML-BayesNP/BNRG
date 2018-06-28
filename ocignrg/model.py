from utils.defs import *
import scipy.sparse
from scipy.stats import poisson
from utils.tpoissrnd import tpoissrnd

def sample_graph(w, V):
    n, C = V.shape
    wb = w.sum()
    hashed_pairs = {}
    for c in range(C):
        u = w*V[:,c]
        ub = u.sum()
        E = npr.poisson(ub*ub/wb)
        pairs = npr.choice(n, size=[E, 2], p=u/ub)
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

def log_likel(graph, w, V):
    wb = w.sum()
    lam = np.outer(w, w)*np.dot(V, V.T)/wb
    ll = -np.triu(lam, k=1).sum()
    row = graph['i']
    col = graph['j']
    lam_pos = (w[row]*w[col]*(V[row]*V[col]).sum(1))/wb
    ll += lam_pos.sum() + log(1 - exp(-lam_pos)).sum()
    return ll

def sample_m(graph, w, V):
    row = graph['i']
    col = graph['j']
    lam = (w[row]*w[col]*(V[row]*V[col]).sum(1))/w.sum()
    return tpoissrnd(lam)

def log_likel_aux(graph, m, w, V):
    n = len(w)
    wb = w.sum()
    ll = 0.5*((w[:,None]*w[:,None]*V*V).sum()/wb \
            - (np.dot(w, V)**2).sum()/wb)
    row = graph['i']
    col = graph['j']
    lam_pos = (w[row]*w[col]*(V[row]*V[col]).sum(1))/wb
    ll += (m*log(lam_pos) - gammaln(m+1)).sum()

    return ll

def log_likel_aux_grad_w(graph, m, w, V):
    n = len(w)
    wb = w.sum()
    wV = np.dot(w, V)
    gw = w*(V**2).sum(1)/wb - 0.5*(w[:,None]*w[:,None]*V*V).sum()/wb**2
    gw -= np.dot(V, wV)/wb - 0.5*(wV**2).sum()/wb**2
    gw += np.bincount(graph['i'], weights=m/(w[graph['i']]+tol), minlength=n)
    gw += np.bincount(graph['j'], weights=m/(w[graph['j']]+tol), minlength=n)
    gw -= m.sum()/(wb + tol)
    return gw

def log_likel_aux_grad_V(graph, m, w, V):
    n = len(w)
    wb = w.sum()
    wV = np.dot(w, V)
    gV = w[:,None]*w[:,None]*V/wb - np.outer(w, wV)/wb
    row = graph['i']
    col = graph['j']
    denom = (V[row]*V[col]).sum(1) + tol
    C = V.shape[1]
    for c in range(C):
        gV[:,c] += np.bincount(row, weights=m*V[col,c]/denom, minlength=n)
        gV[:,c] += np.bincount(col, weights=m*V[row,c]/denom, minlength=n)
    return gV

def sample_w(alpha, beta, n):
    return 1./npr.gamma(alpha, 1./beta, size=n)

def log_prior_w(w, alpha, beta):
    return (alpha*log(beta) - (alpha+1)*log(w) - beta/w - gammaln(alpha)).sum()

def log_prior_grad_w(w, alpha, beta):
    gw = -(alpha+1)/(w + tol) + beta/w**2
    return gw

def sample_V(s, r, n, C):
    return npr.gamma(s, 1./r, size=(n, C))

def log_prior_V(V, s, r):
    return (s*log(r) + (s-1)*log(V) - r*V - gammaln(s)).sum()

def log_prior_grad_V(V, s, r):
    gV = (s-1)/(V + tol) - r
    return gV

def log_prior_grad_s(V, s, r):
    gs = log(r) + log(V) - digamma(s)
    return gs.sum(0)

def log_joint_aux(graph, m, w, V, alpha, beta, s, r):
    return log_likel_aux(graph, m, w, V) \
            + log_prior_w(w, alpha, beta) \
            + log_prior_V(V, s, r)

def log_joint_aux_grad_w(graph, m, w, V, alpha, beta, s, r):
    return log_likel_aux_grad_w(graph, m, w, V) \
            + log_prior_grad_w(w, alpha, beta)

def log_joint_aux_grad_V(graph, m, w, V, alpha, beta, s, r):
    return log_likel_aux_grad_V(graph, m, w, V) \
            + log_prior_grad_V(V, s, r)

def log_joint_aux_uc(graph, m, hw, hV, alpha, beta, s, r):
    return log_joint_aux(graph, m, exp(hw), exp(hV), alpha, beta, s, r) \
            + hw.sum() + hV.sum()

def log_joint_aux_uc_grad_hw(graph, m, hw, hV, alpha, beta, s, r):
    w = exp(hw)
    V = exp(hV)
    gw = log_joint_aux_grad_w(graph, m, w, V, alpha, beta, s, r)
    ghw = gw*w + 1
    return ghw

def log_joint_aux_uc_grad_hV(graph, m, hw, hV, alpha, beta, s, r):
    w = exp(hw)
    V = exp(hV)
    gV = log_joint_aux_grad_V(graph, m, w, V, alpha, beta, s, r)
    ghV = gV*V + 1
    return ghV
