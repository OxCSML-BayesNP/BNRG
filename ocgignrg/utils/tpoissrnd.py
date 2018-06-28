from defs import *
from scipy.stats import poisson

def tpoissrnd(lam):
    x = np.ones(len(lam))
    ind = lam > 1e-5
    if np.any(ind):
        n_ = sum(ind)
        lam_ = lam[ind]
        x[ind] = poisson.ppf(exp(-lam_) + npr.rand(n_)*(1-exp(-lam_)), lam_)
    return x
