from model import *
from utils.plots import *

n = 10000
deg1 = sample_graph(sample_w(-1.0, 0.001, 2.0, n))['deg']
deg2 = sample_graph(sample_w(-1.0, 0.05, 2.5, n))['deg']

plot_degree(deg1, spec='k--', xloglim=8)
plot_degree(deg2, spec='ro-')
plt.xlabel('degree', fontsize=18)
plt.ylabel('distribution', fontsize=18)
plt.tight_layout()
plt.savefig('exponential_cutoff.pdf', bbox_inches='tight')

plt.show()
