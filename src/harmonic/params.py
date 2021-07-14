import numpy as np

from .resonator import convolve
from .resonator import get_sizeranks
from .resonator import params
from .resonator import params_simple


def est_by_docs(fs, doc_fs):
    docsizes = np.array([len(doc_fs[texti]) for texti in doc_fs])
    docweights = np.array([sum(doc_fs[texti].values()) for texti in doc_fs])
    Navg, N = docsizes.dot(docweights) / docweights.sum(), len(fs)
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    theta = 1 - 1 / (1 / np.arange(1, int(N) + 1)).sum()

    return {
        'Navg':  Navg,
        'N':     N,
        'HN':    HN,
        'ravg':  ravg,
        'theta': theta
    }


def regress(rs, fs):
    prebounds = [[1, rs[-1]], [1, sum(fs)], [0.001, 0.999]]
    bounds = [(0, 1), (0, 1), (0.001, 0.999)]
    a, b, theta = params(fs, rs, bounds, prebounds,
                         [rs[-1] / (1 / np.arange(1, rs[-1] + 1)).sum(), rs[-1], np.log10(fs[0]) / np.log10(sum(fs))])
    Navg, N = convolve(a, b, prebounds, invert=True)
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN

    return {
        'Navg':  Navg,
        'N':     N,
        'HN':    HN,
        'ravg':  ravg,
        'theta': theta,
    }


def est_by_theta(fs):
    N = len(fs)
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    theta = 1 - 1 / (1 / np.arange(1, int(N) + 1)).sum()
    Mavg = (fs[0] / sum(fs)) ** -(1 / (1 - theta))
    Navg = (1 - theta) * Mavg

    return {
        'Navg':  Navg,
        'N':     N,
        'HN':    HN,
        'ravg':  ravg,
        'theta': theta,
    }


def regress_by_theta(fs, rs):
    sizeranks = get_sizeranks(fs, rs)

    N = len(fs)
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    srs = np.array([sizeranks[f] for f in fs])
    K1 = params_simple(fs, srs, len(fs) / sum(fs))
    Mavg = sum(fs) * float(K1)
    theta = np.log10(fs[0] * Mavg / sum(fs)) / np.log10(
        Mavg)  # Mavg**theta = fs[0]*Mavg/sum(fs) => Mavg**-(1 - theta) = fs[0]/sum(fs) => Mavg = (fs[0]/sum(fs))**-(1/(1-theta))
    Navg = (1 - theta) * Mavg

    return {
        'Navg':  Navg,
        'N':     N,
        'HN':    HN,
        'ravg':  ravg,
        'theta': theta,
    }


def est_by_types(fs):
    Navg, N = len(fs), sum(fs)
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    theta = 1 - 1 / (1 / np.arange(1, int(N) + 1)).sum()
    return {
        'Navg':  Navg,
        'N':     N,
        'HN':    HN,
        'ravg':  ravg,
        'theta': theta,
    }


def get_model(method='est_type', model_type='mixing', rs=None, fs=None, doc_fs=None):
    assert fs is not None

    rs = np.arange(1, len(fs) + 1)

    if method == 'regress':
        px = regress(rs, fs)
    elif method == 'regress_theta':
        px = regress_by_theta(fs, rs)
    elif method == 'est_doc':
        assert doc_fs is not None
        px = est_by_docs(fs, doc_fs)
    elif method == 'est_theta':
        px = est_by_theta(fs)
    elif method == 'est_type':
        px = est_by_types(fs)
    else:
        raise ValueError('Unrecognized parameter estimation method:', method)

    if model_type == 'simon':
        fmodel = lambda r: ((r - px['theta'])**(-px['theta']))
    elif model_type == 'mixing':
        fmodel = lambda r: ((r - px['theta'])**(-px['theta']))*(1 - (1 + px['ravg']/r)**(-px['Navg']/px['ravg']))
    elif model_type == '?':
        fmodel = lambda r: ((r - px['theta'])**(-px['theta']))*(1 - (1 - ((1/r)/(px['HN']*((px['N'] - px['Navg'])/px['N']) + (1/r)))) * ((1 + px['ravg']/(r - px['theta']))**(-px['Navg']/px['ravg'])))
    else:
        raise ValueError('Unknown model type:', model_type)

    sizeranks = get_sizeranks(fs, rs)
    fhat = fmodel(np.array([sizeranks[f] for f in fs]))
    fnorm = sum(fhat)
    phat = fhat / fnorm

    return fmodel, fhat, fnorm, phat, px
