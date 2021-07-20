import matplotlib.pyplot as plt
import numpy as np

from .base import Action
from .base import BPE

from ..harmonic import get_model
from ..harmonic import get_sizeranks

from ..utils import rankguess
from ..utils import softmax


class ScoredAction(Action):

    def __init__(self, pair, type='merge', count=-1, score=0.0):
        super(ScoredAction, self).__init__(pair, type=type, count=count)
        self.score = float(score)

    def __eq__(self, other):
        return self.score == other.score

    def __gt__(self, other):
        return self.score > other.score


class HRBPE(BPE):

    def __init__(self, tok2ind=None, reg_model='mixing', param_method='est_type', early_stop=False):
        super().__init__(tok2ind=tok2ind)

        self._NLLs = []
        self._Vs = []
        self._MNLLs = []
        self._action_trace = []
        self._recent_actions = set()

        self._start_dist = None
        self._start_model = None
        self._start_params = None

        self._param_method = param_method
        self._reg_model = reg_model
        self._early_stop = early_stop

    def save(self, path, data=None):
        if data is None:
            data = {}

        data['action_trace'] = [[a.pair, 1 if a.type == 'merge' else 0, a.count, a.score] for a in self._action_trace]

        data['reg_model'] = self._reg_model
        data['param_method'] = self._param_method
        data['early_stop'] = self._early_stop

        # data['start_dist'] = [list(self._start_dist[0]), list(self._start_dist[1])]

        super(HRBPE, self).save(path, data=data)

    def load(self, path):
        data = super(HRBPE, self).load(path)

        self._action_trace = [ScoredAction(a[0], count=a[2], score=a[3], type='merge' if a[1] else 'split') for a in data['action_trace']]

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
        
    def init(self, docs, seed=None, method='char'):
        super(HRBPE, self).init(docs, seed=seed, method=method)

        ws, fs = map(np.array, zip(*self._unigraph.most_common()))
        rs = np.array(range(1, len(fs) + 1))
        self._start_dist = np.log10(rs), np.log10(fs / fs.sum())

        fmodel, _, _, _, px = get_model(
            method=self._param_method, model_type=self._reg_model,
            fs=fs, rs=rs, doc_fs=self._doc_unigraph,
        )

        self._start_model = fmodel
        self._start_params = px

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
        Ps = softmax(-np.array([a.score for a in all_actions]) / (fnorm * np.log10(n) * num_characters))
        action_indices = set(np.random.choice(list(range(len(all_actions))), replace=False, p=Ps, size=actions_per_batch))

        return [a for i, a in enumerate(all_actions) if i in action_indices]

    def rank_actions(self, actions):
        ranked = sorted(actions, key=lambda a: a.score)

        # add for logging
        for a in ranked:
            self._MNLLs.append(a.score)
            self._action_trace.append(a)
            self._recent_actions.add(a.pair)

        return ranked

    def do_break_early(self):
        return self._early_stop and len(self._NLLs) > 1 and self._NLLs[-1] > self._NLLs[-2]

    def display_epochs(self):
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

        fig = plt.figure(figsize=(12, 12))

        fig.add_axes([0, 0, 1, 1])
        plt.plot(np.log10(rs), np.log10(self._start_model(rs) / self._start_model(rs).sum()), color='pink', lw=5,
                 label='Starting regularization', linestyle='dashed')
        plt.plot(self._start_dist[0], self._start_dist[1], color='gray', lw=5, label='Starting distribution')
        plt.plot(np.log10(rs), np.log10(fmodel(rs) / fmodel(rs).sum()), label='Converged regularization', color='red',
                 linestyle='dashed', lw=5)
        plt.plot(np.log10(rs), np.log10(fs / fs.sum()), color='black', lw=5,
                 label='Converged HR-BPE (' + str(len(self._NLLs)) + ' iterations)')
        plt.xlabel(r'$\log_{10} r$ Rank', fontsize=25)
        plt.ylabel(r'$\log_{10} p$ Normalized frequency', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim([self._start_dist[1][-1] - 0.1, 0])
        _ = plt.legend(fontsize=25, loc='lower left')

        # fig = plt.figure(figsize = (12,12))

        fig.add_axes([0.5, 0.55, 0.4, 0.4])
        merge_num = range(1, len(self._NLLs) + 1)

        plt.plot(merge_num, self._NLLs, color='black', lw=3, linestyle='dashed',
                 label=r"$\mathcal{E}$")
        plt.xlabel(r'$n$ Epoch number', fontsize=25)
        plt.ylabel(r'$\mathcal{E}$ Normalized entropy', fontsize=25)
        plt_xticks = np.arange(int(min(merge_num)), int(max(merge_num)) + 1, 1.0)[::5] - 1
        plt.xticks(plt_xticks, fontsize=25)
        plt.xlim([min(plt_xticks), max(plt_xticks)])
        plt.yticks(fontsize=25)
        _ = plt.legend(fontsize=25, loc='lower right')

        ax2 = plt.twinx()
        ax2.plot(merge_num, np.log10(self._Vs), color='black', lw=3, linestyle='dotted',
                 label=r"$|V|$")
        ax2.set_ylabel(r'$\log_{10}|V|$ Vocab. size', fontsize=25)
        plt.yticks(fontsize=25)
        _ = plt.legend(fontsize=25, loc='upper right')

        plt.show()
