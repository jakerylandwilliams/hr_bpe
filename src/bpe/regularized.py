import numpy as np
import matplotlib.pyplot as plt
from .base import ScoredAction
from .base import BPE
from ..harmonic import get_model
from ..harmonic import get_sizeranks
from ..utils import rankguess
from ..utils import softmax

# purpose: instantiate a harmonically-regularized bpe model that accepts merges and/or split rules to optimize a harmonic vocabulary
# arguments: (see __init__ method)
# prereqs: (see base.BPE)
# use methods: (see base.BPE). additionally:
# - display_epochs: plot the rank-frequency distribution of the current segmentation of the system's ingested data, as well as a given frequency model and its convergence over optimization epochs
# use attributes: (see base.BPE)
class HRBPE(BPE):
    # purpose: initialize a harmonically-regularized bpe model that accepts merges and/or split rules to optimize a harmonic vocabulary
    # arguments:
    # - tok2ind: (optional) dict, used by .load to set the index
    # - covering_vocab: set, indicating a collection of strs that the tokenizer should consider as bounds for the result of all possible actions
    # - reg_model: str, indicating the type of frequency model to use for regularization, with options from: 
    # -- 'mixing': a simplified/approximate form for the mixing law, modeling an aggregation of multiple preferential selectiond documents
    # -- 'simon': a form considering only the preferential selection mechansim (no document mixing)
    # -- 'resonator':  a complex form for the mixing law, perhaps more accurately modeling an aggregation of multiple preferential selectiond documents
    # - param_method: str, indicating the means by which frequency model parameters will be optimized, with options from: 
    # -- 'regress': utilize gradient-based regression for all model parameters
    # -- 'regress_theta': regress non-theta parameters and determine theta (the replication rate) via ad hoc/expectation-based estimation
    # -- 'est_doc': utilize the known document distribution of ingested data to estimate natural parameters for the model
    # -- 'est_theta': determine theta via an ad hoc/expectation-based estimation and derive other parameters via the mixing law's form
    # - early_stop: bool, with True indicating the model should stop early, i.e., once no actions are predicted to optimize the negative log likelihood
    def __init__(self, tok2ind=None, covering_vocab = set(), reg_model='mixing', param_method='est_type', early_stop=False):
        
        super().__init__(tok2ind=tok2ind, covering_vocab = covering_vocab)

        self._NLLs = []
        self._Vs = []
        self._MNLLs = []
        self._recent_actions = set()

        self._start_dist = None
        self._start_model = None
        self._start_params = None

        self._param_method = param_method
        self._reg_model = reg_model
        self._early_stop = early_stop
        
    # purpose: save a model for later use
    # arguments: (see base.BPE)
    # - path: str, directory location where models are to be saved
    # - data: dict, with fields 'tok2ind', and 'action_trace', which key a dictionary index mapping the vocabulary, and a list of ranked actions to apply as the tokenizers parameters.
    # output: saved model parameters in the location defined by path
    def save(self, path, data=None):
        if data is None:
            data = {}

        data['reg_model'] = self._reg_model
        data['param_method'] = self._param_method
        data['early_stop'] = self._early_stop

        # data['start_dist'] = [list(self._start_dist[0]), list(self._start_dist[1])]

        super(HRBPE, self).save(path, data=data)

    # purpose: load a saved model
    # arguments: (see base.BPE)
    # - path: str, directory location from which the model will be loaded
    # prereqs: a saved model
    # output: loaded model parameters for operation of a trained tokenizer
    def load(self, path):
        data = super(HRBPE, self).load(path)

        self._param_method = data['param_method']
        self._reg_model = data['reg_model']
        self._early_stop = data['early_stop']

        # self._start_dist = (np.array(data['start_dist'][0]), np.array(data['start_dist'][1]))
        #
        # ws, fs = map(np.array, zip(*self._unigraph.most_common()))
        # rs = np.array(range(1, len(fs) + 1))
        # fmodel, _, _, _, px = get_model(
        #     method=self._param_method, model_type=self._reg_model,
        #     fs=fs, rs=rs, doc_fs=self._doc_unigraph,
        # )
        #
        # self._start_model = fmodel
        # self._start_params = px

        return data
        
    # purpose: intialize a HR-BPE model
    # arguments:
    # - docs: list (corpus) of strs (document), containing the data on which the model will be trained
    # - seed: int, indicating the seed of randomization
    # - method: str, one from: 'char' (start from characters), 'warm' (start from a space-based segmentation), or 'rand' (start from a random segmentation)
    # - covering: list (corpus) of lists (documents) of strs (tokens), representing a collection of token boundaries that must be observed during learning, i.e., restricting the learnable rules.
    # - action_protect: list of strs, indicating regular expressions of that cannot be included in actions, protecting the model from, e.g., learning to merge known unwanted tokens
    # prereqs: a corpus of document to either tokenize or initialize for training
    # output: none, data are ingested and structured for learning or application of a model
    def init(self, docs, seed=None, method='char', covering = [], action_protect = ''):
        self._init_method = method
        super(HRBPE, self).init(docs, seed=seed, method=method, covering = covering, action_protect = action_protect)

        ws, fs = map(np.array, zip(*self._unigraph.most_common()))
        rs = np.array(range(1, len(fs) + 1))
        self._start_dist = np.log10(rs), np.log10(fs / fs.sum())

        fmodel, _, _, phat, px = get_model(
            method=self._param_method, model_type=self._reg_model,
            fs=fs, rs=rs, doc_fs=self._doc_unigraph,
        )

        self._start_model = fmodel
        self._start_phat = phat
        self._start_params = px

    # purpose: return a list of actions, ranked according to the contribution of each to the optimization of the harmonic negative log likelihood
    # arguments:
    # - batch_size: int, indicating the number of potentially-optimizing actions to rank per test batch (merge and split, each)
    # - actions_per_batch: int, indicating the number of optimizing actions to sample and test for inclusion as learned rules, per test batch 
    # output: list of ScoredAction objects
    def get_actions(self, batch_size, actions_per_batch):
        ws, fs = map(np.array, zip(*self._unigraph.most_common()))
        rs = np.array(range(1, len(fs) + 1))
        ranks = {w: ix + 1 for ix, w in enumerate(ws)}
        sizeranks = get_sizeranks(fs, rs)

        fmodel, fhat, fnorm, phat, px = get_model(
            method=self._param_method, model_type=self._reg_model,
            fs=fs, rs=rs, doc_fs=self._doc_unigraph,
        )

        fnormed = (fs * np.array([len(w) for w in ws]))
        num_characters = fnormed.sum()
        n = len(ranks)
        m = sum(fs)
        self._NLLs.append(-(fs / m).dot(np.log10(phat)) / np.log10(n))
        self._Vs.append(rs[-1])

        # Query set of split actions to consider
        split_actions = []
        for w, f in list(zip(ws, fs)):
            if len(w) > 1:
                min_act = ScoredAction(('', ''), type='split', count=0, score=float('Inf'))
                for j in range(1, len(w)):
                    pair = (w[:j], w[j:])
                    if pair in self._recent_actions:
                        continue

                    # excess = -1 + int(pair[0] not in ranks) + int(pair[1] not in ranks)
                    fp0 = self._unigraph.get(pair[0], 0)
                    fp1 = self._unigraph.get(pair[1], 0)

                    wdelta = -f * (np.log10(fmodel(rankguess(sizeranks, f + fp0))) +
                                   np.log10(fmodel(rankguess(sizeranks, f + fp1))) +
                                   #  -fp0*np.log10(fmodel(rankguess(sizeranks, fp0))) +
                                   #  -fp1*np.log10(fmodel(rankguess(sizeranks, fp1))) +
                                   -np.log10(fmodel(sizeranks[f])))  # -np.log10(fmodel(ranks[w])))

                    act = ScoredAction(pair, type='split', count=f, score=wdelta)

                    if act < min_act:
                        min_act = act

                if min_act.score < 0:
                    split_actions.append(min_act)
                    if len(split_actions) == batch_size:
                        break

        # Query set of merge actions to consider
        merge_actions = []
        for pair, f in self._digraph.most_common():
            if pair in self._recent_actions:
                continue

            newtok = "".join(pair)
            
            fco = self._unigraph.get(newtok, 0)
            fp0 = self._unigraph.get(pair[0], 0)
            fp1 = self._unigraph.get(pair[1], 0)

            # excess = int(newtok not in ranks) - int(frequency[pair[0]] == f) - int(frequency[pair[1]] == f)
            delta = -f * (np.log10(fmodel(rankguess(sizeranks, f + fco))) +
                          # (fp0-f)*np.log10(fmodel(rankguess(sizeranks, fp0-f))) +
                          # (fp1-f)*np.log10(fmodel(rankguess(sizeranks, fp1-f))) +
                          # -fco*np.log10(fmodel(rankguess(sizeranks, fco))) +
                          -np.log10(fmodel(sizeranks[fp0])) +  # -np.log10(fmodel(ranks[pair[0]])) +
                          -np.log10(fmodel(sizeranks[fp1])))  # -np.log10(fmodel(ranks[pair[1]])))

            if delta < 0:
                merge_actions.append(ScoredAction(pair, count=f, score=delta))
                if len(merge_actions) == batch_size:
                    break

        # sub-sample actions
        all_actions = split_actions + merge_actions
        if all_actions:    
            Ps = softmax(-np.array([a.score for a in all_actions]) / (fnorm * np.log10(n) * num_characters))
            action_indices = set(np.random.choice(list(range(len(all_actions))), replace=False, p=Ps, 
                                                  size=min([actions_per_batch, len(all_actions)])))

            return [a for i, a in enumerate(all_actions) if i in action_indices]
        else:
            return []

    # purpose: rank a list of actions according to the system's current negative log likelihood optimization score for each action's pair of tokens
    # arguments:
    # - actions: list of ScoredAction objects
    # output: list of ScoredAction objects, ordered by decreasing by optimization of likelihood
    def rank_actions(self, actions):
        ranked = sorted(actions, key=lambda a: a.score)

        # add for logging
        for a in ranked:
            self._MNLLs.append(a.score)
            self._recent_actions.add(a.pair)

        return ranked

    # purpose: halt the given training process when no Actions are predicted to improve the negative log likelihood
    # arguments: NA
    # output: boolean, indicating whether or not a stopping criterion has been reached
    def do_break_early(self): 
        return self._early_stop and min(self._NLLs) < self._NLLs[0] and len(self._NLLs) > 1 and self._NLLs[-1] > self._NLLs[-2] 

    # purpose: plot the rank-frequency distribution of the current segmentation of the system's ingested data, as well as a given frequency model and its convergence over optimization epochs
    # arguments:
    # - fname: str (optional) file name where the displayed image will be stored. if left empty, image will NOT be saved to disk
    # prereqs: a trained HR-BPE model with a likelihood optimziation history
    # output: displayed rank-frequency distribution with inset showing optimization as a function of epoch
    def display_epochs(self, fname = ''):
        if self._start_dist is None:
            print(f'Not currently saving starting distribution information...')
            print('Insufficient data to re-create this figure')
            return

        ws, fs = map(np.array, zip(*self._unigraph.most_common()))
        rs = np.arange(1, len(ws) + 1)

        fmodel, fhat, fnorm, phat, px = get_model(
            method=self._param_method, model_type=self._reg_model,
            fs=fs, rs=rs, doc_fs=self._doc_unigraph,
        )

        fig = plt.figure(figsize=(10, 10))

        fig.add_axes([0, 0, 1, 1])
        plt.plot(self._start_dist[0], self._start_dist[1], color='gray', lw=5, label='Starting distribution')
        plt.plot(self._start_dist[0], np.log10(self._start_phat), color='pink', lw=4,
                 label='Starting regularization', linestyle='dashed')
        plt.plot(np.log10(rs), np.log10(fs / fs.sum()), color='black', lw=5,
                 label='Converged HR-BPE (' + str(len(self._NLLs)) + ' iterations)')
        plt.plot(np.log10(rs), np.log10(phat), label='Converged regularization', color='red',
                 linestyle='dashed', lw=4)
        plt.xlabel(r'$\log_{10} r$ Rank', fontsize=20)
        plt.ylabel(r'$\log_{10} p$ Normalized frequency', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([self._start_dist[1][-1] - 0.1, 0])
        _ = plt.legend(fontsize=20, loc='lower left')

        # fig = plt.figure(figsize = (12,12))

        fig.add_axes([0.55, 0.6, 0.35, 0.35])
        merge_num = range(1, len(self._NLLs) + 1)

        plt.plot(merge_num, self._NLLs, color='black', lw=3, linestyle='dashed',
                 label=r"$\mathcal{E}$")
        plt.xlabel(r'$n$ Epoch number', fontsize=20)
        plt.ylabel(r'$\mathcal{E}$ Normalized entropy', fontsize=20)
        plt_xticks = np.arange(int(min(merge_num)), int(max(merge_num)) + 1, 1.0)[::5] - 1
        plt.xticks(plt_xticks, fontsize=25)
        plt.xlim([min(plt_xticks), max(plt_xticks)])
        plt.yticks(fontsize=20)
        _ = plt.legend(fontsize=20, loc='lower right')

        ax2 = plt.twinx()
        ax2.plot(merge_num, np.log10(self._Vs), color='black', lw=3, linestyle='dotted',
                 label=r"$|V|$")
        ax2.set_ylabel(r'$\log_{10}|V|$ Vocab. size', fontsize=20)
        plt.yticks(fontsize=20)
        _ = plt.legend(fontsize=20, loc='upper right')

        if fname:
            plt.savefig(fname, pad_inches=0.1)
        else:
            plt.show()
