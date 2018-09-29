from model.defs import *
from scipy.special import kv
from scipy.optimize import brentq
import ctypes

class TransformedGIG(object):
    def __init__(self, graph=None):
        self.nu = -1 - npr.rand()
        self.hnu = log(-self.nu)
        self.a = 1e-3*npr.rand()
        self.ha = log(self.a)

        if graph is None:
            self.hb = npr.normal()
            self.b = exp(self.hb)
        else:
            f = (lambda x: 0.5*graph['n']*np.sqrt(x)\
                    *kv(self.nu+1, np.sqrt(self.a*x))\
                    /(np.sqrt(self.a)*kv(self.nu, np.sqrt(self.a*x))) \
                    - graph['n_edges'])
            lb = 1e-10
            ub = 10.0
            while f(lb)*f(ub) > 0:
                lb /= 2
                ub *= 2
            self.b = brentq(f, lb, ub)
            self.hb = log(self.b)

    def init_from_chain(self, chain):
        self.nu = np.mean(chain['nu'])
        self.hnu = log(-self.nu)
        self.a = np.mean(chain['a'])
        self.ha = log(self.a)
        self.b = np.mean(chain['b'])
        self.hb = log(self.b)

    @staticmethod
    def sample_(nu, a, b, n=1):
        w = np.zeros(n)
        ctypes.CDLL("utils/gigrnd.so").gigrnd(
                ctypes.c_double(nu),
                ctypes.c_double(a),
                ctypes.c_double(b),
                ctypes.c_int(n),
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return log(w), w

    def sample(self, n=1):
        return TransformedGIG.sample_(self.nu, self.a, self.b, n=n)

    @staticmethod
    def transform(hx):
        return exp(hx)

    @staticmethod
    def inv_transform(x):
        return log(x)

    @staticmethod
    def transform_grad(hx, gx, x=None):
        x = exp(hx) if x is None else x
        return gx*x

    def log_prob(self, hx, x=None, nu=None, a=None, b=None):
        x = self.transform(hx) if x is None else x
        nu = self.nu if nu is None else nu
        a = self.a if a is None else a
        b = self.b if b is None else b
        lp = 0.5*nu*log(a/b) + (nu-1)*hx - 0.5*(a*x + b/x) \
                - log(2*kv(nu, np.sqrt(a*b))) + hx
        return lp.sum()

    def log_prob_grad(self, hx, x=None):
        x = self.transform(hx) if x is None else x
        return self.nu - 0.5*(self.a*x - self.b/x)

    def sample_params(self, hx, x=None):
        hnu_new = npr.normal(self.hnu, 0.05)
        nu_new = -exp(hnu_new)
        log_rho = self.log_prob(hx, x=x, nu=nu_new) \
                - self.log_prob(hx, x=x) \
                -0.5*hnu_new**2 + 0.5*self.hnu**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            self.hnu = hnu_new
            self.nu = nu_new

        ha_new = npr.normal(self.ha, 0.1)
        a_new = exp(ha_new)
        log_rho = self.log_prob(hx, x=x, a=a_new) \
                - self.log_prob(hx, x=x) \
                -0.5*ha_new**2 + 0.5*self.ha**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            self.ha = ha_new
            self.a = a_new

        hb_new = npr.normal(self.hb, 0.05)
        b_new = exp(hb_new)
        log_rho = self.log_prob(hx, x=x, b=b_new) \
                - self.log_prob(hx, x=x) \
                -0.5*hb_new**2 + 0.5*self.hb**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            self.hb = hb_new
            self.b = b_new

    def save_params(self, chain):
        keys = ['nu', 'a', 'b']
        for key in keys:
            if chain.get(key) is None:
                chain[key] = [getattr(self, key)]
            else:
                chain[key].append(getattr(self, key))

    def print_params(self):
        return 'nu {:.4f}, a {:.4f}, b {:.4f}'.format(self.nu, self.a, self.b)
