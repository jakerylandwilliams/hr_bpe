import numpy as np
from .resonator import convolve
from .resonator import get_sizeranks
from .resonator import params
from .resonator import params_simple

# purpose: determine all model parameters via ad hoc estimation based on the rank-frequency distribution and the distribution of document sizes
# arguments:
# - fs: array, sorted (high to low) indicating the frequencies of the systems ingested data (corpus)
# - doc_fs: list (corpus) of Counters (documents) representing the frequencies of tokens in the documents of the ingested data
# output: dict of estimated numerical parameters, keyed as:
# - 'Navg': the estimated size (number of unique types) of a typical document in the represented mixture
# - 'N': the estimated size (number of unique types) of the latent vocabulary being used across the mixture
# - 'HN': pre-computed from 'N', indicating the 'N'th harmonic number 
# - 'ravg': pre-computed from 'N', indicating a typical scale of rank for the latent vocabulary
# - 'theta': the estimated replication rate from the inferred preferential selection process
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

# purpose: determine all model parameters via prediction over a given a rank-frequency distribution
# arguments:
# - fs: array, sorted (high to low) indicating the frequencies of the systems ingested data (corpus)
# - rs: array of integers, pre-computed/equivalent to: np.arange(1,len(fs)+1)
# output:
# - 'Navg': the estimated size (number of unique types) of a typical document in the represented mixture
# - 'N': the estimated size (number of unique types) of the latent vocabulary being used across the mixture
# - 'HN': pre-computed from 'N', indicating the 'N'th harmonic number 
# - 'ravg': pre-computed from 'N', indicating a typical scale of rank for the latent vocabulary
# - 'theta': the estimated replication rate from the inferred preferential selection process
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

# purpose: determine all model parameters via ad hoc estimation, starting from a semantic vocabulary size (N) equal to the number of unique types in the rank-frequency distribution
# arguments:
# - fs: array, sorted (high to low) indicating the frequencies of the systems ingested data (corpus)
# output:
# - 'Navg': the estimated size (number of unique types) of a typical document in the represented mixture
# - 'N': the estimated size (number of unique types) of the latent vocabulary being used across the mixture
# - 'HN': pre-computed from 'N', indicating the 'N'th harmonic number 
# - 'ravg': pre-computed from 'N', indicating a typical scale of rank for the latent vocabulary
# - 'theta': the estimated replication rate from the inferred preferential selection process
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

# purpose: determine some model parameters via prediction (theta and Navg) and the rest (N) from ad hoc rank-frequency analysis
# arguments:
# - fs: array, sorted (high to low) indicating the frequencies of the systems ingested data (corpus)
# - rs: array of integers, pre-computed/equivalent to: np.arange(1,len(fs)+1)
# output:
# - 'Navg': the estimated size (number of unique types) of a typical document in the represented mixture
# - 'N': the estimated size (number of unique types) of the latent vocabulary being used across the mixture
# - 'HN': pre-computed from 'N', indicating the 'N'th harmonic number 
# - 'ravg': pre-computed from 'N', indicating a typical scale of rank for the latent vocabulary
# - 'theta': the estimated replication rate from the inferred preferential selection process
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

# purpose: determine all model parameters via ad hoc estimation, starting from independently setting the semantic vocabulary size (N) and the average document size (Navg) equal to the total number of tokens (sum(fs)) and the number of unique types (len(fs)) in the rank-frequency distribution, respectively
# arguments:
# - fs: array, sorted (high to low) indicating the frequencies of the systems ingested data (corpus)
# output:
# - 'Navg': the estimated size (number of unique types) of a typical document in the represented mixture
# - 'N': the estimated size (number of unique types) of the latent vocabulary being used across the mixture
# - 'HN': pre-computed from 'N', indicating the 'N'th harmonic number 
# - 'ravg': pre-computed from 'N', indicating a typical scale of rank for the latent vocabulary
# - 'theta': the estimated replication rate from the inferred preferential selection process
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

# purpose: gather all information about the model implied by the provided arguments
# arguments:
# - method: str, indicating the means by which frequency model parameters will be optimized, with options from: 
# -- 'regress': utilize gradient-based regression for all model parameters
# -- 'regress_theta': regress non-theta parameters and determine theta (the replication rate) via ad hoc/expectation-based estimation
# -- 'est_doc': utilize the known document distribution of ingested data to estimate natural parameters for the model
# -- 'est_theta': determine theta via an ad hoc/expectation-based estimation and derive other parameters via the mixing law's form
# - model_type: str, indicating the type of frequency model to display, with options from: 
# -- 'mixing': a simplified/approximate form for the mixing law, modeling an aggregation of multiple preferential selectiond documents
# -- 'simon': a form considering only the preferential selection mechansim (no document mixing)
# -- 'resonator':  a complex form for the mixing law, perhaps more accurately modeling an aggregation of multiple preferential selectiond documents
# - rs: array of integers, pre-computed/equivalent to: np.arange(1,len(fs)+1)
# - fs: array, sorted (high to low) indicating the frequencies of the systems ingested data (corpus)
# - doc_fs: list (corpus) of Counters (documents) representing the frequencies of tokens in the documents of the ingested data
# output: tuple with various objects containing model information:
# - fmodel: function of positive integers (ranks), which abstracts model predictions of frequencies for arbitrarily large vocabularies (low frequencies) 
# - fhat: array of predicted frequency values, output by fmodel as a fit of the same size and shape for fs
# - fnorm: float, constant of normaliztion to make fhat into a probabilistic model
# - phat: array of normalized values derived from fhat
# - px: dict of model parameters, keyed as:
# -- 'Navg': the estimated size (number of unique types) of a typical document in the represented mixture
# -- 'N': the estimated size (number of unique types) of the latent vocabulary being used across the mixture
# -- 'HN': pre-computed from 'N', indicating the 'N'th harmonic number 
# -- 'ravg': pre-computed from 'N', indicating a typical scale of rank for the latent vocabulary
# -- 'theta': the estimated replication rate from the inferred preferential selection process
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
    elif model_type == 'resonator':
        fmodel = lambda r: ((r - px['theta'])**(-px['theta']))*(1 - (1 - ((1/r)/(px['HN']*((px['N'] - px['Navg'])/px['N']) + (1/r)))) * ((1 + px['ravg']/(r - px['theta']))**(-px['Navg']/px['ravg'])))
    else:
        raise ValueError('Unknown model type:', model_type)
    sizeranks = get_sizeranks(fs, rs)
    fhat = fmodel(np.array([sizeranks[f] for f in fs]))
    fhat = np.floor(fhat*sum(fs)/sum(fhat)); fhat[fhat==0] = 1
    fnorm = sum(fhat)
    phat = fhat / fnorm
    return fmodel, fhat, fnorm, phat, px