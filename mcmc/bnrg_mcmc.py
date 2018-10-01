import numpy as np
from mcmc.hmc import hmc_step
from utils.ks import compute_ks

def run_mcmc(graph, model, n_samples, burn_in=None, thin=10,
        eps=1e-2, L=20, disp_freq=100, logfile=None):

    n = graph['n']
    model.phw.__init__(graph)
    hw, w = model.phw.sample(n)
    m = model.sample_m(graph, w)
    burn_in = n_samples/4 if burn_in else burn_in
    chain = {}
    chain['w_est'] = np.zeros(n)
    chain['log_joint'] = []
    chain['pred_degree'] = []
    chain['rks'] = []

    burn_in = n_samples/2 if burn_in is None else burn_in
    for t in range(1, n_samples+1):
        # sample hw
        f = lambda hw: (-model.log_joint(graph, m, hw))
        gf = lambda hw: (-model.log_joint_grad(graph, m, hw))
        hw = hmc_step(hw, f, gf, eps, L)
        w = model.phw.transform(hw)
        # sample m
        m = model.sample_m(graph, w)
        # sample hyperparameters
        model.phw.sample_params(hw, x=w)

        if t>burn_in and t%thin==0:
            chain['w_est'] += w
            chain['log_joint'].append(model.log_joint(graph, m, hw))
            model.phw.save_params(chain)
            pred_degree = model.sample_graph(model.phw.sample(n)[1])['deg']
            chain['pred_degree'].append(pred_degree)
            chain['rks'].append(compute_ks(graph['deg'], pred_degree))

        if t%disp_freq == 0:
            line = 'iter {}, log joint {:.4f}, '.format(
                    t, model.log_joint(graph, m, hw))
            line += model.phw.print_params()
            print line
            if logfile is not None:
                logfile.write(line+'\n')

    chain['w_est'] /= len(chain['log_joint'])
    return chain
