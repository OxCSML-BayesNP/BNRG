from utils.hmc import hmc
from model import *
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def run_mcmc(graph, C, n_samples, burn_in=None, thin=10,
        eps=1e-2, L=20, disp_freq=100,
        w=None, nu=None, a=None, b=None):

    # initialize
    n = graph['n']

    if nu is None:
        nu = -1 - npr.rand()
    hnu = log(-nu)

    if a is None:
        a = 1e-3*npr.rand()
    ha = log(a)

    def trans_s(hs):
        s = np.append(hs, 0)
        s = exp(0.5*(s - logsumexp(s)))
        return s

    def inv_trans_s(s):
        return 2*log(s[:-1]) - 2*log(s[-1])

    s = np.ones(C)/np.sqrt(C)
    hs = inv_trans_s(s)
    r = np.ones(C)
    hr = log(r)

    f = (lambda x: 0.5*n*np.sqrt(x)*kv(nu+1, np.sqrt(a*x))/\
            (np.sqrt(a)*kv(nu, np.sqrt(a*x))) \
            - len(graph['i']))
    lb = 1e-10
    ub = 10.0
    while (f(lb)*f(ub) > 0):
        lb /= 2
        ub *= 2
    b = brentq(f, lb, ub)
    hb = log(b)

    if w is None:
        w = sample_w(nu, a, b, n)
    hw = log(w)

    #V = sample_V(s, r, n, C)
    V = np.ones((n, C))/np.sqrt(C)
    hV = log(V)
    m = sample_m(graph, w, V)

    burn_in = n_samples/4 if burn_in is None else burn_in
    chain = {}
    chain['log_joint'] = []
    chain['wV_map'] = np.zeros((n, C))
    chain['V_map'] = np.zeros((n, C))
    chain['nu'] = []
    chain['a'] = []
    chain['b'] = []
    chain['s'] = np.zeros((0, C))
    chain['r'] = np.zeros((0, C))
    lj_max = -np.inf

    for t in range(n_samples):
        # sample hw
        U = lambda hw: (-log_joint_aux_uc(graph, m, hw, hV, nu, a, b, s, r))
        grad_U = lambda hw: (-log_joint_aux_uc_grad_hw(graph, m, hw, hV, nu, a, b, s, r))
        hw, _ = hmc(hw, U, grad_U, eps, L)
        w = exp(hw)

        # sample hV
        U = lambda hV: (-log_joint_aux_uc(graph, m, hw, hV, nu, a, b, s, r))
        grad_U = lambda hV: (-log_joint_aux_uc_grad_hV(graph, m, hw, hV, nu, a, b, s, r))
        hV, _ = hmc(hV, U, grad_U, eps, L)
        V = exp(hV)

        # sample m
        m = sample_m(graph, w, V)

        # sample hyperparameters
        hnu_new = npr.normal(hnu, 0.05)
        nu_new = -exp(hnu_new)
        log_rho = log_prior_w(w, nu_new, a, b) - log_prior_w(w, nu, a, b) \
                -0.5*hnu_new**2 + 0.5*hnu**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hnu = hnu_new
            nu = nu_new

        ha_new = npr.normal(ha, 0.1)
        a_new = exp(ha_new)
        log_rho = log_prior_w(w, nu, a_new, b) - log_prior_w(w, nu, a, b) \
                -0.5*ha_new**2 + 0.5*ha**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            ha = ha_new
            a = a_new

        hb_new = npr.normal(hb, 0.05)
        b_new = exp(hb_new)
        log_rho = log_prior_w(w, nu, a, b_new) - log_prior_w(w, nu, a, b) \
                -0.5*hb_new**2 + 0.5*hb**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hb = hb_new
            b = b_new

        # sample hs
        hs_new = npr.normal(hs, 0.01)
        s_new = trans_s(hs_new)
        r_new = r
        hr_new = hr
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
            lj = log_joint_aux_uc(graph, m, hw, hV, nu, a, b, s, r)
            chain['log_joint'].append(lj)
            if lj > lj_max:
                lj_max = lj
                chain['V_map'] = V
                chain['wV_map'] = w[:,None]*V
            chain['nu'].append(nu)
            chain['a'].append(a)
            chain['b'].append(b)
            chain['s'] = np.append(chain['s'], s[None,:], 0)
            chain['r'] = np.append(chain['r'], r[None,:], 0)

        # display
        if (t+1) % disp_freq == 0:
            print ('iter %d, log_joint %f, nu %f, a %f, b %f') % \
                    (t+1, log_joint_aux_uc(graph, m, hw, hV, nu, a, b, s, r),
                            nu, a, b)
            print 's: ' + str(s)

    return chain
