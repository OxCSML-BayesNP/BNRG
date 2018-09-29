from model.defs import *

# hx -> x = exp(hx)
class TransformedIG(object):
    def __init__(self, graph=None):
        self.halpha = 0.1*npr.normal()
        self.alpha = 1 + exp(self.halpha)
        if graph is None:
            self.hbeta = npr.normal()
            self.beta = exp(self.hbeta)
        else:
            self.beta = 2*(self.alpha-1)*graph['n_edges']/graph['n']
            self.hbeta = log(self.beta)

    def init_from_chain(self, chain):
        self.alpha = np.mean(chain['alpha'])
        self.halpha = log(self.alpha - 1)
        self.beta = np.mean(chain['beta'])
        self.hbeta = log(self.beta)

    @staticmethod
    def sample_(alpha, beta, n=1):
        x = 1./npr.gamma(alpha, 1/beta, n)
        hx = log(x)
        return hx, x

    def sample(self, n=1):
        x = 1./npr.gamma(self.alpha, 1/self.beta, n)
        hx = log(x)
        return hx, x

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

    def log_prob(self, hx, x=None, alpha=None, beta=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        x = exp(hx) if x is None else x
        lp = alpha*log(beta) - (alpha+1)*log(x) \
                - beta/x - gammaln(alpha) + hx
        return lp.sum()

    def log_prob_grad(self, hx, x=None):
        x = exp(hx) if x is None else x
        return -self.alpha + self.beta/(x+tol)

    def sample_params(self, hx, x=None):
        halpha_new = npr.normal(self.halpha, 0.05)
        alpha_new = 1 + exp(halpha_new)
        log_rho = self.log_prob(hx, x=x, alpha=alpha_new) \
                - self.log_prob(hx, x=x) \
                - 0.5*halpha_new**2 + 0.5*self.halpha**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            self.halpha = halpha_new
            self.alpha = alpha_new

        hbeta_new = npr.normal(self.hbeta, 0.01)
        beta_new = exp(hbeta_new)
        log_rho = self.log_prob(hx, x=x, beta=beta_new) \
                - self.log_prob(hx, x=x) \
                - 0.5*hbeta_new**2 + 0.5*self.hbeta**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            self.hbeta = hbeta_new
            self.beta = beta_new

    def save_params(self, chain):
        if chain.get('alpha') is None:
            chain['alpha'] = [self.alpha]
        else:
            chain['alpha'].append(self.alpha)
        if chain.get('beta') is None:
            chain['beta'] = [self.beta]
        else:
            chain['beta'].append(self.beta)

    def print_params(self):
        return 'alpha {:.4f}, beta {:.4f}'.format(self.alpha, self.beta)
