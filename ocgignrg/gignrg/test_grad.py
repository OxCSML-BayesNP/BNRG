from model import *

nu = -1.0
a = 1e-2
b = 2.0
n = 5000
w = sample_w(nu, a, b, n)
graph = sample_graph(w)
m = sample_m(graph, w)
hw = log(w)

lj = log_joint_aux_uc(graph, m, hw, nu, a, b)
ghw = log_joint_aux_uc_grad(graph, m, hw, nu, a, b)

nghw = np.zeros(n)
for i in range(n):
    dhw = np.zeros(n)
    dhw[i] = 1e-5
    nghw[i] = (log_joint_aux_uc(graph, m, hw+dhw, nu, a, b)-lj)/1e-5

print abs((ghw-nghw)/ghw).mean()
