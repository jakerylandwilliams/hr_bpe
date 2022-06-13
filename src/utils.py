import re

import numpy as np

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


def sentokenize(text, space = True, delims = ".?!\n|\t:", sentchars = "a-zA-Z0-9-'"): # harmonize with hr-bpe
    sentences = []
    
    for chunk in re.split("(\s*(?<=["+delims+"][^"+sentchars+"])\s*)", text):
        if (len(chunk)==1 and not re.search("["+sentchars+"]", chunk[0])):
            if space or (chunk[0] != " "):
                if len(sentences):
                    sentences[-1] = sentences[-1] + [chunk]  
                else:
                    sentences.append([chunk])
        elif not re.search("["+sentchars+"]", chunk):
            tokens = tokenize(chunk, space = space)
            if len(sentences):
                sentences[-1] = sentences[-1] + tokens  
            else:
                sentences.append(tokens)
        else:
            sentences.append(tokenize(chunk, space = space))
    
    return sentences


def softmax(z):
    expz = np.exp(z - np.max(z))
    return expz / sum(expz)


def rankguess(sizeranks, f):
    if f in sizeranks:
        return sizeranks[f]
    else:
        sizes = np.array(list(sizeranks.keys()))
        errs = np.abs(f - sizes)
        return sizeranks[sizes[errs == min(errs)][0]]
