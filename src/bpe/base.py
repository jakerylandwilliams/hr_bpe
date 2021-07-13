import json

from abc import ABC
from abc import abstractmethod
from collections import Counter
from collections import defaultdict

import numpy as np

from tqdm import tqdm

from ..utils import tokenize


class Tokenizer(ABC):

    def __init__(self, tok2ind=None, load_path=None):
        if tok2ind is None:
            self._tok2ind = {}
        else:
            self._tok2ind = tok2ind

        if load_path:
            self._tok2ind = json.load(open(load_path))

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

    def save(self, path):
        json.dump(self._tok2ind, open(path, 'w+'))

    @abstractmethod
    def init(self, docs, seed=None):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
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

    def __init__(self, tok2ind=None, load_path=None):
        super().__init__(tok2ind=tok2ind, load_path=load_path)

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

    @abstractmethod
    def init(self, docs, seed=None, method='char'):
        if seed:
            np.random.seed(seed=seed)

        offset = 0
        for doc_idx, doc in tqdm(enumerate(docs), desc=f'Initializing'):

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
            topidx = sorted(set([0] + sorted(np.random.choice(np.arange(1, len(d)), size=int(len(d) / 2), replace=False)) + [len(d)]))
            return [d[topidx[idx - 1]:topidx[idx]] for idx in range(1, len(topidx))]
        else:
            raise ValueError(f'Unrecognized document pre-processing method: {method}')

    @abstractmethod
    def fit(self):
        raise NotImplementedError
