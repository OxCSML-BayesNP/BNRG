import numpy as np
import numpy.random as npr
from scipy.stats import poisson

def tpoissrnd(lam):
    x = np.ones(len(lam))
    ind = lam > 1e-5
    if np.any(ind):
        n_ = sum(ind)
        lam_ = lam[ind]
        x[ind] = poisson.ppf(np.exp(-lam_) \
                + npr.rand(n_)*(1-np.exp(-lam_)), lam_)
    return x
