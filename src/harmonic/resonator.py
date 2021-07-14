from collections import defaultdict

import numpy as np

from scipy.optimize import minimize


def convolve(x, y, prebounds, invert=False):
    if invert:
        a = x
        b = y
        x = int(prebounds[0][0] + a * (prebounds[0][1] - prebounds[0][0]) + 0.5)
        # this constrains the harmonic to be larger
        y = int(prebounds[1][0] + (a + b - a * b) * (prebounds[1][1] - prebounds[1][0]) + 0.5)
        return x, y
    else:
        Edx = (x - prebounds[0][0]) / (prebounds[0][1] - prebounds[0][0])
        Edy = (y - prebounds[1][0]) / (prebounds[1][1] - prebounds[1][0])
        a = Edx
        b = (Edy - Edx) / (1 - Edx)  # this constrains the harmonic to be larger
        return a, b


def resonator(Navg, N, theta, rs):
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    # theta = 1 - 1/(1/np.arange(1,N+1)).sum()
    fhat = ((rs - theta) ** (-theta)) * (1 - (1 + ravg / rs) ** (-Navg / ravg))
    # pw = ((1/rs)/(HN*((N - Navg)/N) + (1/rs)))
    # mixing_model = (1 - (1 - pw) * ((1 + ravg/(rs - theta))**(-Navg/ravg)))
    # harmonic_model = ((rs - theta)**(-theta))
    # fhat = harmonic_model*mixing_model
    # fhat = np.floor(fhat/fhat[-1])
    return fhat


# fmodel = lambda r: ((r - theta)**(-theta))*(1 - (1 - ((1/r)/(HN*((N - Navg)/N) + (1/r)))) * ((1 + ravg/(r - theta))**(-Navg/ravg)))
# fmodel = lambda r: ((r - theta)**(-theta))*(1 - (1 + ravg/r)**(-Navg/ravg))


def NLL(x, fs, rs, bounds, prebounds):
    x = rebound(x, bounds)
    t = x[-1]
    x = tuple(list(convolve(x[0], x[1], prebounds[:2], invert=True)) + [t])
    fhat = resonator(*x, rs)
    # print(x, -(fs*np.log10(fhat/sum(fhat))).sum())
    return -(fs * np.log10(fhat / sum(fhat))).sum()


def rebound(x, bounds):
    ex = list(x)
    for i, bound in enumerate(bounds):
        if x[i] > bound[1]:
            ex[i] = 0.99 * (bound[1] - bound[0]) + bound[0]  # sum(bound)/2 # bound[1]
        elif x[i] < bound[0]:
            ex[i] = 0.01 * (bound[1] - bound[0]) + bound[0]  # sum(bound)/2 # bound[0]
    return tuple(ex)


def resonator_simple(k1, fs, rs):
    HN = (1 / np.arange(1, rs[-1] + 1)).sum()
    N = len(fs)
    ravg = N / HN
    Mavg = sum(fs) * k1
    theta = np.log10(fs[0] * Mavg / sum(fs)) / np.log10(Mavg)
    Navg = (1 - theta) * Mavg
    fhat = ((rs - theta) ** (-theta)) * (1 - (1 + ravg / rs) ** (-Navg / ravg))
    return fhat


def NLL_simple(x, fs, rs, bounds):
    x = rebound([x], bounds)
    fhat = resonator_simple(*x, fs, rs)
    return -(fs * np.log10(fhat / sum(fhat))).sum()


def params_simple(fs, rs, x0):
    bounds = [(0.001, 1)]
    result = minimize(NLL_simple, x0=x0, method='Nelder-Mead', tol=1e-1,
                      options={'maxiter': 999}, args=(fs, rs, bounds))
    return rebound(result.x, bounds)[0]


def params(fs, rs, bounds, prebounds, x0):
    t0 = x0[-1]
    x0 = tuple(list(convolve(x0[0], x0[1], prebounds[:2])) + [t0])
    result = minimize(NLL, x0=x0, method='Nelder-Mead', tol=1e-12,
                      options={'maxiter': 999}, args=(fs, rs, bounds, prebounds))
    return rebound(result.x, bounds)


def get_sizeranks(fs, rs):
    sizeranks = defaultdict(list)
    for f, r in zip(fs, rs):
        sizeranks[f] = max([r, 0 if (f not in sizeranks) else sizeranks[f]])

    return sizeranks
