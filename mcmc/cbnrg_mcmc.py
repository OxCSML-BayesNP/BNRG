import numpy as np
from mcmc.hmc import hmc_step
from utils.ks import compute_ks
from utils.clustering_accuracy import clustering_accuracy

def measure_clustering_accuracy(w, V, labels):
    U = w[:,None]*V
    denom = w.sum()*V.sum(0)/V.shape[0]
    U /= denom
    return clustering_accuracy(np.argmax(U,1), labels)

def run_mcmc(graph, model, n_samples, burn_in=None, thin=10,
        eps=5e-3, L=20, disp_freq=100, logfile=None, w_init=None):

    n = graph['n']
    if w_init is None:
        hw, w = model.phw.sample(n)
    else:
        w = w_init
        hw = model.phw.inv_transform(w)

    c = len(model.phV.gam)
    V = np.ones((n, c))/c
    hV = model.phV.inv_transform(V)

    M = model.sample_M(graph, w, V)
    burn_in = n_samples/4 if burn_in else burn_in
    chain = {}
    chain['w_est'] = np.zeros(n)
    chain['V_est'] = np.zeros((n, c))
    chain['log_joint'] = []
    chain['pred_degree'] = []
    chain['rks'] = []

    burn_in = n_samples/2 if burn_in is None else burn_in
    for t in range(1, n_samples+1):
        # sample hw
        f = lambda hw: (-model.log_joint(graph, M, hw, hV, hw_only=True))
        gf = lambda hw: (-model.log_joint_grad(graph, M, hw, hV, hw_only=True))
        hw = hmc_step(hw, f, gf, eps, L)
        w = model.phw.transform(hw)

        # sample hV
        f = lambda hV: (-model.log_joint(graph, M, hw, hV, hV_only=True))
        gf = lambda hV: (-model.log_joint_grad(graph, M, hw, hV, hV_only=True))
        hV = hmc_step(hV, f, gf, 10*eps, L)
        V = model.phV.transform(hV)

        # sample M
        M = model.sample_M(graph, w, V)
        # sample hyperparams
        model.phw.sample_params(hw, x=w)
        model.phV.sample_params(hV, x=V)

        if t>burn_in and t%thin == 0:
            lj = model.log_joint(graph, M, hw, hV, w=w, V=V)
            chain['w_est'] += w
            chain['V_est'] += V
            chain['log_joint'].append(lj)
            model.phw.save_params(chain)
            _, pred_w = model.phw.sample(n)
            _, pred_V = model.phV.sample(n)
            pred_degree = model.sample_graph(pred_w, pred_V)['deg']
            chain['pred_degree'].append(pred_degree)
            chain['rks'].append(compute_ks(pred_degree, graph['deg']))

        if t%disp_freq == 0:
            line = 'iter {}, log joint {:.4f}, '.format(
                    t, model.log_joint(graph, M, hw, hV))
            line += 'clustering accuracy {:.4f}, '.format(
                    measure_clustering_accuracy(w, V, graph['labels']))
            line += model.phw.print_params()
            print line
            if logfile is not None:
                logfile.write(line+'\n')
    chain['w_est'] /= len(chain['log_joint'])
    chain['V_est'] /= len(chain['log_joint'])

    return chain
