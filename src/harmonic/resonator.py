import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

# purpose: map an arbitrarily-bounded (by prebounds) ordered pair of parameters (x <= y) to [0,1] or invert the mapping
# arguments:
# - x: float, smaller numerical parameter to map to within [0,1]
# - y: float, larger numerical parameter to map to within [0,1]
# - prebounds: list of floats, indicating the range into which x and y both fall
# - invert: boolean, indicating whether the inverse operation (invert = True) should be applied. Note: invert = True changes the convolve's behavior, i.e., it's (x, y)-input should change to two values from within [0,1], the latter of which is larger, namely:
# -- a( = x, when invert=True): float, smaller of the [0,1]-mapped numerical values, to be inverted back into the target range (prebounds)
# -- b( = y, when invert=True): float, larger of the [0,1]-mapped numerical values, to be inverted back into the target range (prebounds)
# output: tuple of values:
# - a (when invert=False): float, smaller of the [0,1]-mapped numerical values, mapped from the target range (prebounds)
# - b (when invert=False): float, larger of the [0,1]-mapped numerical values, mapped from the target range (prebounds)
# -- x (when invert=True): float, smaller numerical parameter to mapped out of [0,1] and into the target range (prebounds)
# -- y (when invert=True): float, larger numerical parameter to mapped out of [0,1] and into the target range (prebounds)
def convolve(x, y, prebounds, invert=False):
    if invert:
        a = x
        b = y
        x = int(prebounds[0][0] + a * (prebounds[0][1] - prebounds[0][0]) + 0.5)
        # this constrains the latter to be larger
        y = int(prebounds[1][0] + (a + b - a * b) * (prebounds[1][1] - prebounds[1][0]) + 0.5)
        return x, y
    else:
        Edx = (x - prebounds[0][0]) / (prebounds[0][1] - prebounds[0][0])
        Edy = (y - prebounds[1][0]) / (prebounds[1][1] - prebounds[1][0])
        a = Edx
        b = (Edy - Edx) / (1 - Edx)  # this constrains the latter to be larger
        return a, b
    
# purpose: reset parameters that violate learning boundaries
# arguments:
# - x: tuple of parameters
# - bounds: list (for each parameter) of tuples of two floats (bounds)
# output: 
# - ex: re-bounded tuple of parameters
def rebound(x, bounds):
    ex = list(x)
    for i, bound in enumerate(bounds):
        if x[i] > bound[1]:
            ex[i] = 0.99 * (bound[1] - bound[0]) + bound[0]  # sum(bound)/2 # bound[1]
        elif x[i] < bound[0]:
            ex[i] = 0.01 * (bound[1] - bound[0]) + bound[0]  # sum(bound)/2 # bound[0]
    return tuple(ex)

# purpose: determine the largest rank attained by each frequency in the rank-frequency distribution
# arguments:
# - fs: array of int frequency values, representing the token occurrences from a corpus
# - rs: array of int rank values, enumerating the sorted frequencies in ascending order
# prereqs:
# output: dictionary mapping integer frequencies to representative integer ranks of largest-possible size
def get_sizeranks(fs, rs):
    sizeranks = defaultdict(list)
    for f, r in zip(fs, rs):
        sizeranks[f] = max([r, 0 if (f not in sizeranks) else sizeranks[f]])
    return sizeranks

# purpose: build a parametric frequency distribution based on the resonator stochastic generation model 
# arguments:
# - Navg: float, the average size (number of unique types) that the resonator will produce for a typical document
# - N: int, the size (number of unique types) of the latent semantic vocabulary used by the resonator
# - theta: float, indicating the preferential selection process replication rate
# - rs: array of integers, pre-computed/equivalent to: np.arange(1,len(fs)+1)
# output:
# - fhat: array of predicted frequency values, of the the same size and shape as rs
def resonator(Navg, N, theta, rs):
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    fhat = ((rs - theta) ** (-theta)) * (1 - (1 + ravg / rs) ** (-Navg / ravg))
    return fhat

# purpose: compute the negative log-10 likelihood of a resonator model, given the data
# arguments:
# - x: tuple of three parameters: Navg, N, and theta (see resonator)
# - fs: array of int frequency values, representing the token occurrences from a corpus
# - rs: array of int rank values, enumerating the sorted frequencies in ascending order
# - bounds: list (for each parameter) of tuples of two floats (bounds) 
# - prebounds: pair of bounds for the N and Navg parameters, for use in mapping to-from [0,1] (see convolve)
# output: float value of the negative log likelihood
def NLL(x, fs, rs, bounds, prebounds):
    x = rebound(x, bounds)
    t = x[-1]
    x = tuple(list(convolve(x[0], x[1], prebounds[:2], invert=True)) + [t])
    fhat = resonator(*x, rs)
    return -(fs * np.log10(fhat / sum(fhat))).sum()

# purpose: utilize sklearn's minimize function to optimize the resonator's parameters
# arguments:
# - fs: array of int frequency values, representing the token occurrences from a corpus
# - rs: array of int rank values, enumerating the sorted frequencies in ascending order
# - bounds: list (for each parameter) of tuples of two floats (bounds) 
# - prebounds: pair of bounds for the N and Navg parameters, for use in mapping to-from [0,1] (see convolve)
# - x0: tuple of values, indicating starting points for optimization of the three parameters: Navg, N, and theta (see resonator)
# output: tuple of values, containing optimized (hopefully) parameters for the resonator model
def params(fs, rs, bounds, prebounds, x0):
    t0 = x0[-1]
    x0 = tuple(list(convolve(x0[0], x0[1], prebounds[:2])) + [t0])
    result = minimize(NLL, x0=x0, method='Nelder-Mead', tol=1e-12,
                      options={'maxiter': 999}, args=(fs, rs, bounds, prebounds))
    return rebound(result.x, bounds)

# purpose: compute a resonator model from ad hoc constrained parameters
# arguments:
# - k1: the reciprocal of the number of documents in the mixture
# - fs: array of int frequency values, representing the token occurrences from a corpus
# - rs: array of int rank values, enumerating the sorted frequencies in ascending order
# output:
# - fhat: array of predicted frequency values, of the the same size and shape as rs
def resonator_simple(k1, fs, rs):
    HN = (1 / np.arange(1, rs[-1] + 1)).sum()
    N = len(fs)
    ravg = N / HN
    Mavg = sum(fs) * k1
    theta = np.log10(fs[0] * Mavg / sum(fs)) / np.log10(Mavg)
    Navg = (1 - theta) * Mavg
    fhat = ((rs - theta) ** (-theta)) * (1 - (1 + ravg / rs) ** (-Navg / ravg))
    return fhat

# purpose: compute the negative log-10 likelihood of a 'simple' (ad hoc constrained) resonator model, given the data
# arguments:
# - x: tuple of one parameter: k1 (see resonator_simple)
# - fs: array of int frequency values, representing the token occurrences from a corpus
# - rs: array of int rank values, enumerating the sorted frequencies in ascending order
# output: float value of the negative log likelihood
def NLL_simple(x, fs, rs, bounds):
    x = rebound([x], bounds)
    fhat = resonator_simple(*x, fs, rs)
    return -(fs * np.log10(fhat / sum(fhat))).sum()

# purpose: utilize sklearn's minimize function to optimize the resonator's parameters when the k-document mixture-constraint is applied
# arguments:
# - fs: array of int frequency values, representing the token occurrences from a corpus
# - rs: array of int rank values, enumerating the sorted frequencies in ascending order
# - x0: tuple of one value, indicating starting points for the reciprocal of the number of documents in the mixture (k1, see resonator_simple)
# output: tuple of values, containing optimized (hopefully) parameters for the resonator model
def params_simple(fs, rs, x0):
    bounds = [(0.001, 1)]
    result = minimize(NLL_simple, x0=x0, method='Nelder-Mead', tol=1e-1,
                      options={'maxiter': 999}, args=(fs, rs, bounds))
    return rebound(result.x, bounds)[0]
