from model.defs import *

# hx -> z = sigmoid(hx) -> x ~ dirichlet(gam)
class TransformedDir(object):
    def __init__(self, c, gam0=1.0):
        self.gam = gam0*np.ones(c)
        self.hgam = log(self.gam)

    @staticmethod
    def get_beta_params(gam):
        a = gam.cumsum()[::-1][1:]
        b = gam[:-1]
        return a, b

    @staticmethod
    def beta_transform(z):
        n = len(z)
        return np.c_[1-z, np.ones((n,1))] * np.c_[np.ones((n,1)), z.cumprod(1)]

    @staticmethod
    def sample_(gam, n=1):
        a, b = TransformedDir.get_beta_params(gam)
        z = npr.beta(a, b, size=[n, len(gam)-1])
        x = TransformedDir.beta_transform(z)
        hx = TransformedDir.inv_transform(x)
        return hx, x

    def sample(self, n=1):
        return TransformedDir.sample_(self.gam, n=n)

    @staticmethod
    def transform(hx):
        z = 1/(1 + exp(-hx))
        return TransformedDir.beta_transform(z)

    @staticmethod
    def inv_transform(x):
        n = x.shape[0]
        denom = np.c_[np.ones((n,1)), 1-x.cumsum(1)[:,:-2]] + tol
        z = (1 - x[:,:-1]/denom).clip(0., 1.)
        return log(z) - log(1-z)

    @staticmethod
    def transform_grad(hx, gx, x=None):
        x = self.transform(hx) if x is None else x
        z = 1/(1 + exp(-hx))
        ghx = (x[:,:-1]/(z-1+tol))*gx[:,:-1] \
                + (x*gx)[:,::-1].cumsum(1)[:,::-1][:,1:]/z
        ghx *= z*(1-z)
        return ghx

    def log_prob(self, hx, x=None, gam=None):
        gam = self.gam if gam is None else gam
        a, b = TransformedDir.get_beta_params(gam)
        z = 1/(1 + exp(-hx))
        lp = gammaln(a + b) - gammaln(a) - gammaln(b) \
                + a*log(z) + b*log(1-z)
        return lp.sum()

    def log_prob_grad(self, hx, x=None):
        a, b = TransformedDir.get_beta_params(self.gam)
        z = 1/(1 + exp(-hx))
        return a*(1-z) - b*z

    def sample_params(self, hx, x=None):
        hgam_new = npr.normal(self.hgam, 0.01)
        gam_new = exp(hgam_new)
        log_rho = self.log_prob(hx, x=x, gam=gam_new) \
                - self.log_prob(hx, x=x) \
                - 0.5*(hgam_new**2).sum() + 0.5*(self.hgam**2).sum()
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            self.hgam = hgam_new
            self.gam = gam_new

if __name__ == '__main__':

    gam = np.array([0.1, 0.4, 0.5, 0.01])
    pV = TransformedDir(gam)
    x = pV.sample(100000)
    print gam/gam.sum()
    print x.mean(0)

    #x = np.array([[2.69482029e-04, 9.95819157e-01, 3.91136134e-03, 6.48171907e-20]])
    #print x.sum()
    #print inv_transform(x)
