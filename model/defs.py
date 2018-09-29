import numpy as np
import numpy.random as npr
from scipy.special import gammaln, digamma
from scipy.misc import logsumexp

tol = 1e-20
exp = np.exp
log = lambda x: np.log(x + tol)
