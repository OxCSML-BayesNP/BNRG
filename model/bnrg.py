from model.defs import *
from utils.tpoissrnd import tpoissrnd

class BNRG(object):
    def __init__(self, phw):
        self.phw = phw

    @staticmethod
    def sample_graph(w):
        n = len(w)
        wb = w.sum()
        n_edges = npr.poisson(wb)
        pairs = npr.choice(n, size=[n_edges, 2], p=w/wb)
        unique_pairs = {}
        for (i, j) in pairs:
            if i < j and unique_pairs.get((i, j)) is None:
                unique_pairs[(i, j)] = 1
        graph = {}
        graph['n'] = n
        graph['n_edges'] = len(unique_pairs)
        graph['i'] = np.zeros(len(unique_pairs), dtype=int)
        graph['j'] = np.zeros(len(unique_pairs), dtype=int)
        graph['deg'] = np.zeros(n, dtype=int)
        for k, (i, j) in enumerate(unique_pairs):
            graph['i'][k] = i
            graph['j'][k] = j
            graph['deg'][i] += 1
            graph['deg'][j] += 1
        return graph

    @staticmethod
    def sample_m(graph, w):
        Lam_pos = w[graph['i']]*w[graph['j']]/w.sum()
        return tpoissrnd(Lam_pos)

    def log_likel(self, graph, m, hw, w=None):
        w = self.phw.transform(hw) if w is None else w
        n = len(w)
        wb = w.sum()
        ll = 0.5*((w**2).sum()/wb - wb)
        wi = w[graph['i']]
        wj = w[graph['j']]
        ll += (m*(log(wi) + log(wj) - log(wb))).sum()
        return ll

    def log_likel_grad(self, graph, m, hw, w=None):
        w = self.phw.transform(hw) if w is None else w
        n = len(w)
        wb = w.sum()
        gw = w/wb - 0.5*(w**2/wb**2).sum() - 0.5
        gw += np.bincount(graph['i'], weights=m/(w[graph['i']]+tol), minlength=n)
        gw += np.bincount(graph['j'], weights=m/(w[graph['j']]+tol), minlength=n)
        gw -= m.sum()/(wb + tol)
        return self.phw.transform_grad(hw, gw, x=w)

    def log_joint(self, graph, m, hw, w=None):
        return (self.log_likel(graph, m, hw, w=w) \
                + self.phw.log_prob(hw, x=w))

    def log_joint_grad(self, graph, m, hw, w=None):
        return (self.log_likel_grad(graph, m, hw, w=w) \
                + self.phw.log_prob_grad(hw, x=w))
