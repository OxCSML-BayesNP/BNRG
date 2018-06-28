from utils.hmc import hmc
from model import *

def run_mcmc(graph, C, n_samples, burn_in=None, thin=10,
        eps=1e-2, L=20, disp_freq=100,
        w=None, alpha=None, beta=None):

    def mm_beta(w, alpha):
        x = (exp(-1./w).mean())**(1./alpha)
        return x/(1-x)

    # initialize
    n = graph['n']
    if alpha is None:
        halpha = 0.1*npr.normal()
        alpha = 1 + exp(halpha)
    else:
        halpha = log(alpha-1)

    def trans_s(hs):
        s = np.append(hs, 0)
        s = exp(0.5*(s - logsumexp(s)))
        return s

    def inv_trans_s(s):
        return 2*(log(s[:-1]) - log(s[-1]))

    s = np.ones(C)/np.sqrt(C)
    hs = inv_trans_s(s)
    r = np.ones(C)
    hr = log(r)

    if beta is None:
        beta = 2*(alpha-1)*len(graph['i'])/n
        hbeta = log(beta)
    else:
        hbeta = log(beta)

    if w is None:
        w = sample_w(alpha, beta, n)
    hw = log(w)
    V = sample_V(s, r, n, C)
    hV = log(V)
    m = sample_m(graph, w, V)

    burn_in = n_samples/4 if burn_in is None else burn_in
    chain = {}
    chain['log_joint'] = []
    chain['wV_map'] = np.zeros((n, C))
    chain['V_map'] = np.zeros((n, C))
    chain['alpha'] = []
    chain['beta'] = []
    chain['s'] = np.zeros((0, C))
    chain['r'] = np.zeros((0, C))
    lj_max = -np.inf

    for t in range(n_samples):
        # sample hw
        U = lambda hw: (-log_joint_aux_uc(graph, m, hw, hV, alpha, beta, s, r))
        grad_U = lambda hw: (-log_joint_aux_uc_grad_hw(graph, m, hw, hV, alpha, beta, s, r))
        hw, _ = hmc(hw, U, grad_U, eps, L)
        w = exp(hw)

        # sample hV
        U = lambda hV: (-log_joint_aux_uc(graph, m, hw, hV, alpha, beta, s, r))
        grad_U = lambda hV: (-log_joint_aux_uc_grad_hV(graph, m, hw, hV, alpha, beta, s, r))
        hV, _ = hmc(hV, U, grad_U, eps, L)
        V = exp(hV)

        # sample m
        m = sample_m(graph, w, V)

        # sample hyperparameters
        halpha_new = npr.normal(halpha, 0.05)
        alpha_new = 1 + exp(halpha_new)
        log_rho = log_prior_w(w, alpha_new, beta) \
                - log_prior_w(w, alpha, beta) \
                - 0.5*halpha_new**2 + 0.5*halpha**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            halpha = halpha_new
            alpha = alpha_new

        #hbeta_mean = log(mm_beta(w, alpha))
        #hbeta_new = npr.normal(hbeta_mean, 0.05)
        #beta_new = exp(hbeta_new)
        #log_rho = log_prior_w(w, alpha, beta_new)-log_prior_w(w, alpha, beta)\
        #        - 0.5*hbeta_new**2 + 0.5*hbeta**2 \
        #        - 0.5*(hbeta-hbeta_mean)**2/(0.05**2) \
        #        + 0.5*(hbeta_new-hbeta_mean)**2/(0.05**2)
        hbeta_new = npr.normal(hbeta, 0.05)
        beta_new = exp(hbeta_new)
        log_rho = log_prior_w(w, alpha, beta_new) \
                - log_prior_w(w, alpha, beta) \
                - 0.5*hbeta_new**2 + 0.5*hbeta**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hbeta = hbeta_new
            beta = beta_new

        # sample hs
        hs_new = npr.normal(hs, 0.05)
        s_new = trans_s(hs_new)
        hr_new = hr
        r_new = exp(hr_new)
        #s_new = exp(hs_new)/np.sqrt(C)
        #r_new = s_new*np.sqrt(C)
        #hr_new = npr.normal(hr, 0.01)
        #r_new = exp(hr_new)
        log_rho = log_prior_V(V, s_new, r_new) - log_prior_V(V, s, r) \
                - 0.5*(hs_new**2).sum() + 0.5*(hs**2).sum() \
                - hs.sum() + hs_new.sum()
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hs = hs_new
            s = s_new
            hr = hr_new
            r = r_new

        # save samples
        if (t+1) > burn_in and (t+1)%thin == 0:
            lj = log_joint_aux_uc(graph, m, hw, hV, alpha, beta, s, r)
            chain['log_joint'].append(lj)
            if lj > lj_max:
                lj_max = lj
                chain['wV_map'] = w[:,None]*V
            chain['alpha'].append(alpha)
            chain['beta'].append(beta)
            chain['s'] = np.append(chain['s'], s[None,:], 0)
            chain['r'] = np.append(chain['r'], r[None,:], 0)
            chain['V_map'] = V

        # display
        if (t+1) % disp_freq == 0:
            print ('iter %d, log_joint %f, alpha %f, beta %f') % \
                    (t+1, log_joint_aux_uc(graph, m, hw, hV, alpha, beta, s, r),
                            alpha, beta)
            print 's: ' + str(s)

    return chain
