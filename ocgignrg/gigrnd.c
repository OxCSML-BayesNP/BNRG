// This code is a c version of the MATLAB code,
// https://uk.mathworks.com/matlabcentral/fileexchange/53594-gigrnd-p--a--b--samplesize-
// which implements the algorithm presented in Devroye 2014,
// L. Devroye, Random variate generation for the generalized inverse Gaussian distribution,
// Statistics and Computing, Vol. 24, pp. 239-246, 2014.

#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MIN(a, b) (((a)<(b))?(a):(b))

double randu(void)
{
    return rand() / (RAND_MAX + 1.0);
}

double psi(double x, double alpha, double lambda)
{
    return -alpha*(cosh(x)-1) - lambda*(exp(x) - x - 1);
}

double dpsi(double x, double alpha, double lambda)
{
    return -alpha*sinh(x) - lambda*(exp(x) - 1);
}

double g(double x, double sd, double td, double f1, double f2)
{
    double a = 0;
    double b = 0;
    double c = 0;
    if ((x >= -sd) && (x <= td)) a = 1;
    else if (x > td) b = f1;
    else if (x < -sd) c = f2;
    return a + b + c;
}

void gigrnd(double nu, double a, double b, int n, double *x)
{
    srand(time(NULL));
    double lambda = nu;
    double omega = sqrt(a*b);
    int swap = 0;
    if (lambda < 0) {
        lambda = -lambda;
        swap = 1;
    }
    double alpha = sqrt(omega*omega + lambda*lambda) - lambda;

    // find t
    double t;
    double y = -psi(1, alpha, lambda);
    if ((y >= 0.5) && (y <= 2)) t = 1;
    else if (y > 2) t = sqrt(2/(alpha+lambda));
    else if (y < 0.5) t = log(4/(alpha + 2*lambda));

    // find s
    double s;
    y = -psi(-1, alpha, lambda);
    if ((y >= 0.5) && (y <= 2)) s = 1;
    else if (y > 2) s = sqrt(4/(alpha*cosh(1) + lambda));
    else if (y < 0.5) s = MIN(1/lambda, log(1+1/alpha+sqrt(1/(alpha*alpha)+2/alpha)));

    // generation
    double eta = -psi(t, alpha, lambda);
    double zeta = -dpsi(t, alpha, lambda);
    double theta = -psi(-s, alpha, lambda);
    double xi = dpsi(-s, alpha, lambda);
    double p = 1/xi;
    double r = 1/zeta;
    double td = t - r*eta;
    double sd = s - p*theta;
    double q = td + sd;
    int i;
    double u, v, w, f1, f2;
    for (i=0; i < n; ++i) {
        while (1) {
            u = randu();
            v = randu();
            w = randu();
            if (u < (q/(p + q + r))) x[i] = -sd + q*v;
            else if (u < ((q+r)/(p+q+r))) x[i] = td - r*log(v);
            else x[i] = -sd + p*log(v);
            f1 = exp(-eta - zeta*(x[i]-t));
            f2 = exp(-theta + xi*(x[i]+s));
            if (w*g(x[i], sd, td, f1, f2) <= exp(psi(x[i], alpha, lambda)))
                break;
        }
        x[i] = exp(x[i]) * (lambda/omega + sqrt(1 + lambda*lambda/(omega*omega)));
        if (swap) x[i] = 1/x[i];
        x[i] /= sqrt(a/b);
    }
    return;
}
