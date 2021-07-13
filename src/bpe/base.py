import json

from abc import ABC
from abc import abstractmethod


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
    def init(self, docs):
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
