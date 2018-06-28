import numpy as np
import numpy.random as npr
import scipy
from scipy.special import gammaln, digamma
from scipy.misc import logsumexp

tol = 1e-40
exp = np.exp
#log = lambda x: np.log(x + tol)
log = np.log
