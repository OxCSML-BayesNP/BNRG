import numpy as np
from mcmc.hmc import hmc_step
from utils.ks import compute_ks
from utils.clustering_accuracy import clustering_accuracy

def compute_labels(w, V):
    U = w[:,None]*V
    denom = w.sum()*V.sum(0)/V.shape[0]
    U /= denom
    return np.argmax(U, 1)

def measure_clustering_accuracy(w, V, labels):
    return clustering_accuracy(compute_labels(w, V), labels)

def run_mcmc(graph, model, n_samples, n_init=None, burn_in=None,
        thin=10, eps=5e-3, L=20, disp_freq=100, logfile=None):

    n = graph['n']
    model.phw.__init__(graph)
    c = len(model.phV.gam)
    model.phV.__init__(c)
    V = np.ones((n, c))/c
    hV = model.phV.inv_transform(V)
    n_init = n_samples/10 if n_init is None else n_init
    if n_init is not None:
        line = 'initializing V...'
        print line
        logfile.write(line+'\n')
        w = np.ones(n)
        hw = model.phw.inv_transform(w)
        M = model.sample_M(graph, w, V)
        for t in range(1, n_init+1):
            # sample hV
            f = lambda hV: (-model.log_joint(graph, M, hw, hV, hV_only=True))
            gf = lambda hV: (-model.log_joint_grad(graph, M, hw, hV, hV_only=True))
            hV = hmc_step(hV, f, gf, 1e-1, L)
            V = model.phV.transform(hV)

            # sample M
            M = model.sample_M(graph, w, V)

            # sample hyperparams
            model.phV.sample_params(hV, V)

            if t % disp_freq==0:
                line = 'iter {}, log joint {:.4f}, '.format(
                        t, model.log_joint(graph, M, hw, hV))
                line += 'clustering accuracy {:.4f}, '.format(
                    measure_clustering_accuracy(w, V, graph['labels']))
                print line
                if logfile is not None:
                    logfile.write(line+'\n')

    hw, w = model.phw.sample(n)
    burn_in = n_samples/4 if burn_in else burn_in
    chain = {}
    chain['max_lj'] = -np.inf
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
        V_eps = 5*eps if t < burn_in else eps
        hV = hmc_step(hV, f, gf, V_eps, L)
        V = model.phV.transform(hV)

        # sample M
        M = model.sample_M(graph, w, V)

        # sample hyperparams
        model.phw.sample_params(hw, w)
        model.phV.sample_params(hV, V)

        if t>burn_in and t%thin == 0:
            lj = model.log_joint(graph, M, hw, hV, w=w, V=V)
            if lj > chain['max_lj']:
                chain['w_est'] = w
                chain['V_est'] = V
                chain['max_lj'] = lj
            chain['log_joint'].append(lj)
            model.phw.save_params(chain)
            _, pred_w = model.phw.sample(n)
            _, pred_V = model.phV.sample(n)
            pred_degree = model.sample_graph(pred_w, pred_V)['deg']
            chain['pred_degree'].append(pred_degree)
            chain['rks'].append(compute_ks(graph['deg'], pred_degree))

        if t%disp_freq == 0:
            line = 'iter {}, log joint {:.4f}, '.format(
                    t, model.log_joint(graph, M, hw, hV))
            line += 'clustering accuracy {:.4f}, '.format(
                    measure_clustering_accuracy(w, V, graph['labels']))
            line += model.phw.print_params()
            print line
            if logfile is not None:
                logfile.write(line+'\n')

    return chain
