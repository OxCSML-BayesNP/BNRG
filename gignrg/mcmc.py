from utils.hmc import hmc
from model import *
from scipy.optimize import brentq
import os
import pickle

def run_mcmc(graph, n_samples, burn_in=None, thin=10,
        eps=1e-2, L=20, disp_freq=100, return_w=False):

    # initialize
    n = graph['n']

    nu = -1 - npr.rand()
    hnu = log(-nu)

    a = 1e-3*npr.rand()
    ha = log(a)

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

    w = sample_w(nu, a, b, n)
    hw = log(w)
    m = sample_m(graph, w)

    burn_in = n_samples/4 if burn_in is None else burn_in
    chain = {}
    chain['log_joint'] = []
    chain['nu'] = []
    chain['a'] = []
    chain['b'] = []

    for t in range(n_samples):
        # sample hw
        U = lambda hw: (-log_joint_aux_uc(graph, m, hw, nu, a, b))
        grad_U = lambda hw: (-log_joint_aux_uc_grad(graph, m, hw, nu, a, b))
        hw, _ = hmc(hw, U, grad_U, eps, L)
        w = exp(hw)

        # sample m
        m = sample_m(graph, w)

        # sample hypeparams
        hnu_new = npr.normal(hnu, 0.05)
        nu_new = -exp(hnu_new)
        log_rho = log_prior(w, nu_new, a, b) - log_prior(w, nu, a, b) \
                -0.5*hnu_new**2 + 0.5*hnu**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hnu = hnu_new
            nu = nu_new

        ha_new = npr.normal(ha, 0.1)
        a_new = exp(ha_new)
        log_rho = log_prior(w, nu, a_new, b) - log_prior(w, nu, a, b) \
                -0.5*ha_new**2 + 0.5*ha**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            ha = ha_new
            a = a_new

        hb_new = npr.normal(hb, 0.05)
        b_new = exp(hb_new)
        log_rho = log_prior(w, nu, a, b_new) - log_prior(w, nu, a, b) \
                -0.5*hb_new**2 + 0.5*hb**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hb = hb_new
            b = b_new

        # save samples
        if (t+1) > burn_in and (t+1)%thin == 0:
            chain['log_joint'].append(log_joint_aux_uc(graph, m, hw, nu, a, b))
            chain['nu'].append(nu)
            chain['a'].append(a)
            chain['b'].append(b)

        # display
        if (t+1) % disp_freq == 0:
            print ('iter %d, log_joint %f, nu %f, a %f, b %f' % \
                    (t+1, log_joint_aux_uc(graph, m, hw, nu, a, b),
                        nu, a, b))

    if return_w:
        return chain, w
    else:
        return chain
