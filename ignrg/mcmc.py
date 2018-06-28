from utils.hmc import hmc
from model import *
import os
import pickle

def run_mcmc(graph, n_samples, burn_in=None, thin=10,
        eps=1e-2, L=20, disp_freq=100, return_w=False):

    # initialize
    n = graph['n']
    halpha = 0.1*npr.normal()
    alpha = 1 + exp(halpha)

    beta = 2*(alpha-1)*len(graph['i'])/n
    hbeta = log(beta)

    w = sample_w(alpha, beta, n)
    hw = log(w)
    m = sample_m(graph, w)

    burn_in = n_samples/4 if burn_in is None else burn_in
    chain = {}
    chain['log_joint'] = []
    chain['alpha'] = []
    chain['beta'] = []

    for t in range(n_samples):
        # sample hw
        U = lambda hw: (-log_joint_aux_uc(graph, m, hw, alpha, beta))
        grad_U = lambda hw: (-log_joint_aux_uc_grad(graph, m, hw, alpha, beta))
        hw, _ = hmc(hw, U, grad_U, eps, L)
        w = exp(hw)

        # sample m
        m = sample_m(graph, w)

        # sample/estimate hyperparameters
        halpha_new = npr.normal(halpha, 0.05)
        alpha_new = 1 + exp(halpha_new)
        log_rho = log_prior(w, alpha_new, beta) - log_prior(w, alpha, beta) \
                - 0.5*halpha_new**2 + 0.5*halpha**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            halpha = halpha_new
            alpha = alpha_new

        hbeta_new = npr.normal(hbeta, 0.01)
        beta_new = exp(hbeta_new)
        log_rho = log_prior(w, alpha, beta_new) - log_prior(w, alpha, beta) \
                - 0.5*hbeta_new**2 + 0.5*hbeta**2
        rho = exp(min(log_rho, 0.0))
        if npr.rand() < rho:
            hbeta = hbeta_new
            beta = beta_new

        # save samples
        if (t+1) > burn_in and (t+1)%thin == 0:
            chain['log_joint'].append(log_joint_aux_uc(graph, m, hw, alpha, beta))
            chain['alpha'].append(alpha)
            chain['beta'].append(beta)

        # display
        if (t+1) % disp_freq == 0:
            print ('iter %d, log_joint %f, alpha %f, beta %f') % \
                    (t+1, log_joint_aux_uc(graph, m, hw, alpha, beta), alpha, beta)

    if return_w:
        return chain, w
    else:
        return chain
