# copied from https://stackoverflow.com/a/33346366
import numpy as np

def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def fill_ecdf(quantiles, cumprob):
    lb = min(quantiles)
    ub = max(quantiles)
    n = ub-lb + 1
    filled_quantiles = np.arange(lb, ub+1)
    filled_cumprob = np.zeros(n)
    pointer = -1
    for i in range(n):
        if quantiles[pointer+1] == filled_quantiles[i]:
            pointer += 1
        filled_cumprob[i] = cumprob[pointer]
    return filled_quantiles, filled_cumprob

def compute_ks(deg1, deg2, normalize=False):
    q1, p1 = ecdf(deg1)
    q1, p1 = fill_ecdf(q1, p1)
    q2, p2 = ecdf(deg2)
    q2, p2 = fill_ecdf(q2, p2)

    lb = max(q1[0], q2[0])
    ub = min(q1[-1], q2[-1])
    p1 = p1[np.logical_and(q1>=lb, q1<=ub)]
    p2 = p2[np.logical_and(q2>=lb, q2<=ub)]
    if normalize:
        norm = np.sqrt(p2*(1-p2))
        ind = norm > 0
        D = abs(p1[ind]-p2[ind])/norm[ind]
    else:
        D = abs(p1-p2)
    return D.max()

if __name__ == '__main__':

    q = np.array([3, 4, 6, 8, 10])
    p = np.array([0.3, 0.4, 0.6, 0.8, 1.0])
    fq, fp = fill_ecdf(q, p)
    print fq
    print fp
