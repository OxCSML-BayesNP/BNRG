from defs import *

def hmc(x0, U, grad_U, eps, L):
    x, r = x0.copy(), npr.normal(size=x0.shape)
    H0 = U(x) + 0.5*(r*r).sum()
    for _ in range(L):
        r -= 0.5*eps*grad_U(x)
        x += eps*r
        r -= 0.5*eps*grad_U(x)
    H = U(x) + 0.5*(r*r).sum()
    rho = exp(min(H0-H, 0.0))
    if npr.random() < rho:
        return x, rho
    else:
        return x0.copy(), rho

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = np.array([[0.1, 0.3]])
    A = np.array([[2.25, 1.2], [1.2, 1.25]])
    iA = np.linalg.inv(A)
    U = lambda x: 0.5*(np.dot(x-a, iA)*(x-a)).sum(axis=1)
    #grad_U = grad(U)
    grad_U = lambda x: np.dot(x-a, iA)

    ### visualize
    x1, x2 = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
    y = exp(-U(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape))
    plt.contour(x1, x2, y)

    eps = 0.1
    L = 20
    N = 3000
    x = npr.normal(size=[1,2])
    for i in range(1, N):
        x_, _ = hmc(x[-1][None,:], U, grad_U, eps, L)
        x = np.r_[x, x_]
    plt.scatter(x[:,0], x[:,1], color='red', alpha=0.5)
    plt.show()

    print x.mean(axis=0)
    print np.cov(x.T)
