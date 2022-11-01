import re
import numpy as np

# purpose: segment a text (str object) into a sequence of smaller str tokens
# arguments:
# - text: str, representing the document (text) to be tokenized
# - space: bool, with False indicating that space characters (" ") will be filtered from output
# - wordchars: str, indicating the contents of a character class that will represent the range of characters used by the tokenizer for 'word'-like objects
# prereqs: a str object (text) for tokenization
# output: a list (document) of strs (tokens)
def tokenize(text, space = True, wordchars = "a-zA-Z0-9-'"):
    tokens = []
    for token in re.split("(["+wordchars+"]+)", text):
        if not space:
            token = re.sub("[ ]+", "", token)
        if not token:
            continue
        if re.search("["+wordchars+"]", token):
            tokens.append(token)
        else: 
            tokens.extend(token)
    return tokens

# purpose: segment a text (str object) into a sequence of sentence-sequences of smaller str tokens
# arguments:
# - text: str, representing the document (text) to be tokenized
# - space: bool, with False indicating that space characters (" ") will be filtered from output
# - delims: str, indicating the contents of a character class that will represent the range of characters used by the sentokenizer for 'delimiter'-like objects
# - sentchars: str, indicating the contents of a character class that will represent the range of characters used by the tokenizer for 'word'-like objects within the sentences
# prereqs: a str object (text) for tokenization
# output: a list (document) of lists (sentences) of strs (tokens)
def sentokenize(text, space = True, delims = ".?!\n|\t:", sentchars = "a-zA-Z0-9-'"): 
    sentences = []
    for chunk in re.split("(\s*(?<=["+delims+"][^"+sentchars+"])\s*)", text):
        if (len(chunk)==1 and not re.search("["+sentchars+"]", chunk[0])):
            if space or (chunk[0] != " "):
                if len(sentences):
                    sentences[-1] = sentences[-1] + [chunk]  
                else:
                    sentences.append([chunk])
        elif not re.search("["+sentchars+"]", chunk):
            tokens = tokenize(chunk, space = space, wordchars = sentchars)
            if len(sentences):
                sentences[-1] = sentences[-1] + tokens  
            else:
                sentences.append(tokens)
        else:
            sentences.append(tokenize(chunk, space = space, wordchars = sentchars))
    return sentences

# purpose: compute the softmax probability distribution for a given array of real numbers
# arguments:
# - z: array of floats
# output: an array of positive floats that sum to 1
def softmax(z):
    expz = np.exp(z - np.max(z))
    return expz / sum(expz)

# purpose: guess a rank for the 
# arguments:
# - sizeranks: dict of int values (ranks), keyed by int value frequencies
# - f: float test frequency for which a rank will be guessed (if f not in sizeranks)
# output: a rank that approximates (at least) the one assigned to a token of the same frequency in the given vocabulary('s sizeranks)
def rankguess(sizeranks, f):
    if f in sizeranks:
        return sizeranks[f]
    else:
        sizes = np.array(list(sizeranks.keys()))
        errs = np.abs(f - sizes)
        return sizeranks[sizes[errs == min(errs)][0]]

# purpose: determine character index-locations for the starting characters of each token in a sequence of tokens
# arguments:
# - tokens: list (sentence) of strs (tokens)
# output: a list of ints, indicating the starting character of each token in tokens
def get_spans(tokens):
        locs = [0] + list(np.cumsum([len(t) for t in tokens]))
        return  list(zip(locs[0:-1],locs[1:]))

# purpose: determine a positive performance evaluation of a given segmentation against a gold standard (correctly found boundaries)
# arguments:
# - ts: list (sentence) of strs (tokens), representing the gold standard tokenization
# - ts_hat: list (sentence) of strs (tokens), representing the predicted tokenization
# output: tuple of floats, representing the precision, recall, and F1-score of the predicted segmentation
def eval_segmentation(ts, ts_hat):
    y = set(get_spans(ts)); y_hat = set(get_spans(ts_hat))
    TP = len(y_hat.intersection(y)); FP = len(y_hat - y); FN = len(y - y_hat)
    P = TP/(TP+FP) if (TP+FP) else 0
    R = TP/(TP+FN) if (TP+FN) else 0
    F1 = 2*P*R/(P+R) if (P+R) else 0
    return P, R, F1