import json

from abc import ABC
from abc import abstractmethod
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from ..harmonic import get_model

from ..utils import tokenize


class Action:

    def __init__(self, pair, type='merge', count=-1):
        self.pair = pair
        self.type = type
        self.count = count


class Tokenizer(ABC):

    def __init__(self, tok2ind=None):
        if tok2ind is None:
            self._tok2ind = {}
        else:
            self._tok2ind = tok2ind

        self._ind2tok = {v: k for k, v in self._tok2ind.items()}

    def __len__(self):
        return len(self._tok2ind)

    def add_type(self, tok):
        if tok not in self._tok2ind:
            self._tok2ind[tok] = len(self._tok2ind)
            self._ind2tok[self._tok2ind[tok]] = tok

    def del_type(self, tok):
        if tok in self._tok2ind:
            idx = self._tok2ind[tok]

            del self._ind2tok[idx]
            del self._tok2ind[tok]

            # shifting down each type that's a larger index
            # than the one just removed
            i = idx + 1
            while i in self._ind2tok:
                t = self._ind2tok[i]
                self._tok2ind[t] = i - 1
                self._ind2tok[i - 1] = t

                del self._ind2tok[i]

    def save(self, path, data=None):
        if data is None:
            data = {}

        data['tok2ind'] = self._tok2ind
        json.dump(data, open(path, 'w+'))

    def load(self, path):
        data = json.load(open(path))

        self._tok2ind = data['tok2ind']
        self._ind2tok = {v: k for k, v in self._tok2ind.items()}

        return data

    @abstractmethod
    def init(self, docs, seed=None):
        raise NotImplementedError

    @abstractmethod
    def fit(self, num_batches, batch_size=1, seed=None):
        raise NotImplementedError

    def encode(self, text):
        return self.tokens_to_indices(self.tokenize(text))

    def tokenize(self, text, start=-1):
        for ix in range(len(self._tok2ind) - 1 if start < 0 else start, -1, -1):
            tok = self._ind2tok[ix]
            if text == tok:
                return [tok]
            elif tok in text:
                segs = text.split(tok)
                enc = []
                for s in segs:
                    if s:
                        enc.extend(self.tokenize(s, start=ix - 1))
                    enc.append(tok)
                return enc[:-1]

        return []

    def decode(self, indices):
        return ''.join(self.indices_to_tokens(indices))

    def tokens_to_indices(self, toks):
        return [self._tok2ind[t] for t in toks]

    def indices_to_tokens(self, indices):
        return [self._ind2tok[i] for i in indices]


class BPE(Tokenizer):

    def __init__(self, tok2ind=None):
        super().__init__(tok2ind=tok2ind)

        # starting and ending points for each token (as a set for constant lookup)
        self._lefts = {}
        self._rights = {}

        # frequency-based information
        self._unigraph = Counter()
        self._doc_unigraph = defaultdict(Counter)
        self._digraph = Counter()

        # mapping to and from indices
        self._tok_idx = defaultdict(set)
        self._pair_idx = defaultdict(set)
        self._char2docidx = {}

    def save(self, path, data=None):
        if data is None:
            data = {}

        data['unigraph'] = dict(self._unigraph)
        data['digraph'] = [[k[0], k[1], v] for k, v in self._digraph.items()]
        data['doc_unigraph'] = {k: dict(v) for k, v in self._doc_unigraph.items()}
        
        super(BPE, self).save(path, data=data)

    def load(self, path):
        data = super(BPE, self).load(path)

        self._unigraph = Counter(data['unigraph'])
        self._digraph = Counter({(l, r): v for l, r, v in data['digraph']})
        self._doc_unigraph = defaultdict(Counter)
        for k, v in data['doc_unigraph'].items():
            self._doc_unigraph[k] = Counter(v)

        return data

    def init(self, docs, seed=None, method='char'):
        if seed:
            np.random.seed(seed=seed)

        offset = 0
        for doc_idx, doc in enumerate(tqdm(docs, desc=f'Initializing')):

            stream = self._init_doc(doc, method=method)
            assert (sum(map(len, stream)) == len(doc))

            for ix, tok in enumerate(stream):
                self._unigraph[tok] += 1
                self._tok_idx[tok].add(offset)

                for char_idx in range(offset, offset + len(tok)):
                    self._char2docidx[char_idx] = doc_idx

                self._doc_unigraph[doc_idx][tok] += 1

                tok_pair = (stream[ix - 1], tok) if ix else ('', tok)
                self._lefts[(offset - len(stream[ix - 1])) if ix else (offset - 1)] = tok_pair
                self._rights[offset] = tok_pair

                if ix:
                    self._digraph[tok_pair] += 1
                    self._pair_idx[tok_pair].add(offset - len(stream[ix - 1]))

                offset += len(tok)

            tok_pair = (tok, '')
            self._lefts[offset - len(tok)] = tok_pair
            self._rights[offset] = tok_pair

            offset += 1

    @staticmethod
    def _init_doc(d, method='char'):
        if method == 'char':
            return d
        elif method == 'warm':
            return tokenize(d)
        elif method == 'rand':
            topidx = sorted(set(
                [0] + sorted(np.random.choice(np.arange(1, len(d)), size=int(len(d) / 2), replace=False)) + [len(d)]))
            return [d[topidx[idx - 1]:topidx[idx]] for idx in range(1, len(topidx))]
        else:
            raise ValueError(f'Unrecognized document pre-processing method: {method}')

    def fit(self, num_batches, batch_size=1, seed=None):
        if seed:
            np.random.seed(seed=seed)

        for batch in tqdm(range(num_batches), desc='Fitting'):
            actions = self.rank_actions(self.get_actions(batch_size))[:batch_size]

            for action in actions:
                if action.type == 'merge':
                    self.merge(action.pair)
                else:
                    self.split(action.pair)

            if self.do_break_early():
                break

        for k in self._unigraph.keys():
            self.add_type(k)

        print(f'Built a vocabulary of {len(self)} types')

    def merge(self, pair):
        newtok = "".join(pair)

        skip_next = False
        locations = list(self._pair_idx[pair])
        for i in sorted(locations):
            if skip_next:  # handle odd numbers of repeated tokens
                skip_next = False
                continue

            # gather the instance's neighbors
            lneighbor = self._rights[i][0]
            rneighbor = self._lefts[i + len(pair[0])][1]
            skip_next = True if pair[0] == pair[1] and pair[1] == rneighbor else False

            # delete the entries for this pair in both indices
            del (self._lefts[i])
            del (self._rights[i + len(pair[0])])

            # gather the old left and right adjacent pairs
            lpair = (lneighbor, pair[0])
            rpair = (pair[1], rneighbor)

            # construct new left and right adjacent pairs
            newlpair = (lneighbor, newtok)
            newrpair = (newtok, rneighbor)

            # delete the old left and right pair from both left/right indexings
            del (self._lefts[i - len(lneighbor) if lneighbor else i - 1])  # lpair
            del (self._rights[i])  # lpair
            del (self._lefts[i + len(pair[0])])  # rpair
            del (self._rights[i + len(newtok)])  # rpair

            # update both left and right indexings with the new left and right pairs
            self._lefts[i - len(lneighbor) if lneighbor else i - 1] = newlpair
            self._rights[i] = newlpair
            self._lefts[i] = newrpair
            self._rights[i + len(newtok)] = newrpair

            # only update left co-occurrences if lneighbor is non-empty
            if lneighbor:  # including deleting the lpair instance from codata
                self._digraph[newlpair] += 1
                self._digraph[lpair] -= 1
                self._pair_idx[newlpair].add(i - len(lneighbor))
                self._pair_idx[lpair].remove(i - len(lneighbor))
                if not self._digraph[lpair]:
                    del (self._digraph[lpair])
                if not self._pair_idx[lpair]:
                    del (self._pair_idx[lpair])

            # only update right co-occurrences if rneighbor is non-empty
            if rneighbor:  # including deleting rpair the instance from codata
                self._digraph[newrpair] += 1
                self._digraph[rpair] -= 1
                self._pair_idx[newrpair].add(i)
                self._pair_idx[rpair].remove(i + len(pair[0]))
                if not self._digraph[rpair]:
                    del (self._digraph[rpair])
                if not self._pair_idx[rpair]:
                    del (self._pair_idx[rpair])

            # update unigram frequencies
            self._unigraph[newtok] += 1
            self._unigraph[pair[0]] -= 1
            self._unigraph[pair[1]] -= 1
            if not self._unigraph[pair[0]]:
                del (self._unigraph[pair[0]])
            if not self._unigraph[pair[1]]:
                del (self._unigraph[pair[1]])

            texti = self._char2docidx[i]
            self._doc_unigraph[texti][newtok] += 1
            self._doc_unigraph[texti][pair[0]] -= 1
            if not self._doc_unigraph[texti][pair[0]]:
                del (self._doc_unigraph[texti][pair[0]])
            self._doc_unigraph[texti][pair[1]] -= 1
            if not self._doc_unigraph[texti][pair[1]]:
                del (self._doc_unigraph[texti][pair[1]])

            # update the token locations
            self._tok_idx[newtok].add(i)
            self._tok_idx[pair[0]].remove(i)
            self._tok_idx[pair[1]].remove(i + len(pair[0]))
            if not self._tok_idx[pair[0]]:
                del (self._tok_idx[pair[0]])
            if not self._tok_idx[pair[1]]:
                del (self._tok_idx[pair[1]])

            # delete the pair from the co-occurrence data record
            self._digraph[pair] -= 1
            self._pair_idx[pair].remove(i)
            if not self._pair_idx[pair]:
                del (self._pair_idx[pair])
            if not self._digraph[pair]:
                del (self._digraph[pair])

    def split(self, wpair):
        oldtok = "".join(wpair)
        locations = list(self._tok_idx[oldtok])
        for i in sorted(locations):
            # update the left/right and consequential digraph indices
            # wpair[0] updates
            lneighbor = self._rights[i][0]
            rneighbor = self._lefts[i][1]
            lpair = (lneighbor, oldtok)
            rpair = (oldtok, rneighbor)
            newlpair = (lneighbor, wpair[0])
            newcpair = wpair
            newrpair = (wpair[1], rneighbor)

            # cpair
            self._digraph[newcpair] += 1
            self._pair_idx[newcpair].add(i)
            self._lefts[i] = wpair
            self._rights[i + len(wpair[0])] = wpair

            # lpairs
            del (self._rights[i])
            self._rights[i] = newlpair
            del (self._lefts[i - len(lneighbor) if lneighbor else i - 1])
            self._lefts[i - len(lneighbor) if lneighbor else i - 1] = newlpair
            if lneighbor:
                self._digraph[newlpair] += 1
                self._digraph[lpair] -= 1
                self._pair_idx[newlpair].add(i - len(lneighbor))
                self._pair_idx[lpair].remove(i - len(lneighbor))
                if not self._digraph[lpair]:
                    del self._digraph[lpair]
                if not self._pair_idx[lpair]:
                    del (self._pair_idx[lpair])

            # rpairs
            # del(left_indexed_pairs[i]) # technically, this was just overwritten w/wpair and doesn't need deletion
            self._lefts[i + len(wpair[0])] = newrpair
            # del(right_indexed_pairs[i+len(oldtok)])
            self._rights[i + len(oldtok)] = newrpair
            if rneighbor:
                self._digraph[newrpair] += 1
                self._digraph[rpair] -= 1
                self._pair_idx[newrpair].add(i + len(wpair[0]))
                self._pair_idx[rpair].remove(i)
                if not self._digraph[rpair]:
                    del (self._digraph[rpair])
                if not self._pair_idx[rpair]:
                    del (self._pair_idx[rpair])

            # update unigram frequencies
            self._unigraph[oldtok] -= 1
            self._unigraph[wpair[0]] += 1
            self._unigraph[wpair[1]] += 1
            if not self._unigraph[oldtok]:
                del self._unigraph[oldtok]

            # update the token locations
            self._tok_idx[oldtok].remove(i)
            self._tok_idx[wpair[0]].add(i)
            self._tok_idx[wpair[1]].add(i + len(wpair[0]))
            if not self._tok_idx[oldtok]:
                del (self._tok_idx[oldtok])

            texti = self._char2docidx[i]
            self._doc_unigraph[texti][oldtok] -= 1
            if not self._doc_unigraph[texti][oldtok]:
                del (self._doc_unigraph[texti][oldtok])
            self._doc_unigraph[texti][wpair[0]] += 1
            self._doc_unigraph[texti][wpair[1]] += 1

    @abstractmethod
    def get_actions(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def rank_actions(self, actions):
        raise NotImplementedError

    def do_break_early(self):
        return False

    def display(self, model_type='mixing', method='est_type'):
        ws, fs = map(np.array, zip(*self._unigraph.most_common()))
        rs = np.arange(1, len(ws) + 1)

        fmodel, fhat, fnorm, phat, px = get_model(
            method=method, model_type=model_type,
            fs=fs, rs=rs, doc_fs=self._doc_unigraph,
        )

        plt.plot(np.log10(rs), np.log10(fs / fs.sum()), color='black', lw=3, label='BPE frequency')
        plt.plot(np.log10(rs), np.log10(fmodel(rs) / fmodel(rs).sum()), label='Regularization',
                 color='red', linestyle='dashed', lw=3)
        plt.xlabel(r'$\log_{10} r$ rank', fontsize=25)
        plt.ylabel(r'$\log_{10} f$ frequency', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        _ = plt.legend(fontsize=25)

        plt.show()
