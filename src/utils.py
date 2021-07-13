import re


def tokenize(text, space=True):
    tokens = []
    for token in re.split("([0-9a-zA-Z'-]+)", text):
        if not space:
            token = re.sub("[ ]+", "", token)

        if not token:
            continue

        if re.search("[0-9a-zA-Z'-]", token):
            tokens.append(token)
        else:
            tokens.extend(token)

    return tokens


def sentokenize(text, space=True):
    sentences = []
    for sentence in re.split("(\s+(?<=[.?!,;:\n][^a-zA-Z0-9])\s*)", text):
        if len(sentence) == 1 and not re.search("[0-9a-zA-Z'-]", sentence[0]):
            if len(sentences):
                sentences[-1] = sentences[-1] + [sentence]
            else:
                sentences.append([sentence])
        elif not re.search("[0-9a-zA-Z'-]", sentence):
            tokens = tokenize(sentence, space=space)
            if len(sentences):
                sentences[-1] = sentences[-1] + tokens
            else:
                sentences.append(tokens)
        else:
            sentences.append(tokenize(sentence, space=space))
    return sentences
