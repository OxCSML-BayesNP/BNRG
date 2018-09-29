from defs import *
from utils.tpoissrnd import tpoissrnd
from utils.mnrnd import mnrnd

class CBNRG(object):
    def __init__(self, phw, phV):
        self.phw = phw
        self.phV = phV

    @staticmethod
    def sample_graph(w, V):
        n, c = V.shape
        U = w[:,None]*V
        denom = w.sum()*V.sum(0)/n
        unique_pairs = {}
        for k in range(c):
            Ub = U[:,k].sum()
            n_edges = npr.poisson(Ub**2/denom[k])
            pairs = npr.choice(n, size=[n_edges,2],
                    p=exp(log(U[:,k]) - log(Ub)))
            for (i, j) in pairs:
                if i < j and unique_pairs.get((i, j)) is None:
                    unique_pairs[(i, j)] = 1
        graph = {}
        graph['n'] = n
        graph['n_edges'] = len(unique_pairs)
        graph['i'] = np.zeros(len(unique_pairs), dtype=int)
        graph['j'] = np.zeros(len(unique_pairs), dtype=int)
        graph['deg'] = np.zeros(n, dtype=int)
        for l, (i, j) in enumerate(unique_pairs):
            graph['i'][l] = i
            graph['j'][l] = j
            graph['deg'][i] += 1
            graph['deg'][j] += 1
        return graph

    @staticmethod
    def sample_M(graph, w, V):
        U = w[:,None]*V
        Ub = U.sum(0)
        Lam_pos = (U[graph['i']]*U[graph['j']]/Ub)
        denom = Lam_pos.sum(1)
        n_trials = tpoissrnd(denom).astype(np.int)
        return mnrnd(n_trials, Lam_pos/denom[:,None]).astype(np.float)

    def log_likel(self, graph, M, hw, hV, w=None, V=None):
        w = self.phw.transform(hw) if w is None else w
        V = self.phV.transform(hV) if V is None else V
        U = w[:,None]*V
        denom = w.sum()*V.sum(0)/V.shape[0]
        ll = 0.5*((U**2).sum(0)/denom - (U.sum(0))**2/denom).sum()
        Lam_pos = (U[graph['i']]*U[graph['j']])/denom
        ll += (M*log(Lam_pos) - gammaln(M+1)).sum()
        return ll

    def log_likel_naive(self, graph, M, hw, hV, w=None, V=None):
        w = self.phw.transform(hw) if w is None else w
        V = self.phV.transform(hV) if V is None else V
        U = w[:,None]*V
        denom = w.sum()*V.sum(0)/V.shape[0]
        n, c = V.shape
        M_ = np.zeros((n, n, c))
        for l, (i,j) in enumerate(zip(graph['i'], graph['j'])):
            M_[i,j] = M[l]
        ll = 0
        for i in range(n-1):
            for j in range(i+1, n):
                for k in range(c):
                    lam = w[i]*w[j]*V[i,k]*V[j,k]/denom[k]
                    ll -= lam
                    if M_[i,j,k] > 0:
                        ll += M_[i,j,k]*log(lam)
        return ll

    def log_likel_grad(self, graph, M, hw, hV, w=None, V=None,
            hw_only=False, hV_only=False):
        both = (not hw_only) and (not hV_only)
        w = self.phw.transform(hw) if w is None else w
        V = self.phV.transform(hV) if V is None else V
        n, c = V.shape
        wb = w.sum()
        Vb = V.sum(0)
        denom = wb*V.sum(0)/n
        U = w[:,None]*V
        row = graph['i']
        col = graph['j']

        if hw_only or both:
            Mb = M.sum(1)
            gw = (U*V/denom).sum(1) - (V*U.sum(0)/denom).sum(1)
            gw += 0.5*(((U.sum(0))**2 - (U**2).sum(0))/(wb*denom)).sum()
            gw += np.bincount(row, weights=Mb/(w[row]+tol), minlength=n)
            gw += np.bincount(col, weights=Mb/(w[col]+tol), minlength=n)
            gw -= Mb.sum()/(wb + tol)
            ghw = self.phw.transform_grad(hw, gw, w)

        if hV_only or both:
            gV = (w[:,None]*U - w[:,None]*U.sum(0, keepdims=True))/denom
            gV += 0.5*(((U.sum(0))**2 - (U**2).sum(0))/(Vb*denom))
            for k in range(c):
                gV[:,k] += np.bincount(row, weights=M[:,k]/(V[row,k]+tol), minlength=n)
                gV[:,k] += np.bincount(col, weights=M[:,k]/(V[col,k]+tol), minlength=n)
                gV[:,k] -= (M[:,k]/(Vb[k]+tol)).sum()
            ghV = self.phV.transform_grad(hV, gV, V)

        if hw_only:
            return ghw
        elif hV_only:
            return ghV
        else:
            return ghw, ghV

    def log_joint(self, graph, m, hw, hV, w=None, V=None,
            hw_only=False, hV_only=False):
        lj = self.log_likel(graph, m, hw, hV, w=w, V=V)
        if hw_only:
            lj += self.phw.log_prob(hw, w)
            return lj
        elif hV_only:
            lj += self.phV.log_prob(hV, V)
            return lj
        else:
            lj += self.phw.log_prob(hw, w) + self.phV.log_prob(hV, V)
            return lj

    def log_joint_grad(self, graph, m, hw, hV, w=None, V=None,
            hw_only=False, hV_only=False):
        grad = self.log_likel_grad(graph, m, hw, hV, w=w, V=V,
                hw_only=hw_only, hV_only=hV_only)
        if hw_only:
            ghw = grad
            ghw += self.phw.log_prob_grad(hw, w)
            return ghw
        elif hV_only:
            ghV = grad
            ghV += self.phV.log_prob_grad(hV, V)
            return ghV
        else:
            ghw, ghV = grad
            ghw += self.phw.log_prob_grad(hw, w)
            ghV += self.phV.log_prob_grad(hV, V)
            return ghw, ghV

from model.transformed_ig import TransformedIG
from model.transformed_dir import TransformedDir
def test_grad():
    n = 1000
    c = 4
    alpha = 1.5
    beta = 2.0
    gam = np.ones(c)

    hw, w = TransformedIG.sample_(alpha, beta, n)
    hV, V = TransformedDir.sample_(gam, n)
    phw = TransformedIG()
    phw.alpha = alpha
    phw.beta = beta
    phV = TransformedDir(c)
    phV.gam = gam
    graph = CBNRG.sample_graph(w, V)
    model = CBNRG(phw, phV)
    M = CBNRG.sample_M(graph, w, V)

    target = 'log_joint'
    f = getattr(model, target)
    gf = getattr(model, target+'_grad')
    y0 = f(graph, M, hw, hV)
    ghw, ghV = gf(graph, M, hw, hV)

    nghw = np.zeros(n)
    for i in range(n):
        dhw = np.zeros(n)
        dhw[i] = 1e-5
        nghw[i] = (f(graph, M, hw+dhw, hV)-y0)/1e-5
    print abs(ghw-nghw).mean()/abs(ghw).mean()

    nghV = np.zeros((n, c-1))
    for i in range(n):
        for k in range(c-1):
            dhV = np.zeros((n, c-1))
            dhV[i,k] = 1e-5
            nghV[i,k] = (f(graph, M, hw, hV+dhV)-y0)/1e-5
    print abs(ghV-nghV).mean()/abs(ghV).mean()

if __name__=='__main__':
    test_grad()
