import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.bpe import GreedyBPE
from src.bpe import HRBPE

def train_model(docs, seed, train_handle, method = "hr-bpe", init_method = "char", num_batches = 100, batch_size = 1_000, actions_denom = 1, language = "EN", train_ex = -1, reg_model = "mixing", param_method = "regress_theta"):
    actions_per_batch = int(batch_size / actions_denom)

    if method == "greedy":
        model = GreedyBPE()
    elif method == "hr-bpe":
        model = HRBPE()
    else:
        raise ValueError
    
    if train_ex == -1:
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{language}-{train_handle}'
    elif train_ex > 0:
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{language}-{train_handle}-{train_ex}'
    else:
        raise ValueError

    try:
        model.load(cache_dir / (model_str + ".json"))
    except FileNotFoundError:
        model.init(docs, seed = seed, method = init_method)
        model.fit(num_batches, batch_size, actions_per_batch = actions_per_batch, seed = seed)

        model.save(cache_dir / (model_str + ".json"))

    return model

def read_training_data(train_sets, data_dir, language, seed, n):
    train_handle = '-'.join(train_sets)
    docs = ["".join(ts) for train_set in train_sets for ts in json.load(open(data_dir / language / (train_set + ".json"))).values()]
    if n > 0:
        random.seed(seed)
        docs = random.sample(docs, n)
    elif n == -1:
        pass
    else:
        raise ValueError
    return docs, train_handle

def get_spans(tokens):
        locs = [0] + list(np.cumsum([len(t) for t in tokens]))
        return  list(zip(locs[0:-1],locs[1:]))

def eval_segmentation(ts, ts_hat):
        y = set(get_spans(ts)); y_hat = set(get_spans(ts_hat))
        TP = len(y_hat.intersection(y)); FP = len(y_hat - y); FN = len(y - y_hat)
        P = TP/(TP+FP) if (TP+FP) else 0
        R = TP/(TP+FN) if (TP+FN) else 0
        F1 = 2*P*R/(P+R) if (P+R) else 0
        return P, R, F1

def eval_dataset(test_set, data_dir, language, model):
    dataset = json.load(open(data_dir / language / (test_set + ".json")))

    Ps, Rs, F1s = [], [], []
    for rix, ts in enumerate(tqdm(list(dataset.values()))):
        if not ts:
            continue
        ts_hat = model.tokenize(''.join(ts))
        P, R, F1, = eval_segmentation(ts, ts_hat)
        Ps.append(P); Rs.append(R); F1s.append(F1)
    
    print(test_set, ": P: ", np.mean(Ps), "R: ",  np.mean(Rs), "F1: ", np.mean(F1s))

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, help = "dataset location", default = "./data/gold")
parser.add_argument("--cache_dir", type = str, help = "cache location", default = "./cache")
parser.add_argument("--language", type = str, help = "dataset language", default = "EN")
parser.add_argument("--train_sets", type = str, nargs = "+", help = "list of training set names, stored in ./[data_dir]/[language]/", default = ["ewtb", "lowlands", "ritter", "parseme-train"])
parser.add_argument("--test_sets", type = str, nargs = "+", help = "list of test set names, stored in ./[data_dir]/[language]/", default = ["tweebank", "trustpilot", "ted", "parseme-test"])
parser.add_argument("--method", type = str, help = "'greedy' or 'hr-bpe'", default = "hr-bpe")
parser.add_argument("--init_method", type = str, help = "'char', 'warm', or 'rand'", default = "char")
parser.add_argument("--seed", type = int, default = 42)
parser.add_argument("--train_ex", type = int, help = "number of examples to train on", default = -1)
parser.add_argument("--num_batches", type = int, default = 100)
parser.add_argument("--batch_size", type = int, default = 1_000)
parser.add_argument("--actions_denom", type = int, default = 1)
parser.add_argument("--reg_model", type = str, help = "'simon', 'mixing', or 'resonator'", default = "mixing")
parser.add_argument("--regress_theta", type = str, help = "'est_type', 'est_doc', 'est_theta', 'regress', or 'regress_theta'", default = "regress_theta")

args = parser.parse_args()

data_dir = Path(args.data_dir)
cache_dir = Path(args.cache_dir)

docs, train_handle = read_training_data(args.train_sets, data_dir, args.language, args.seed, args.train_ex)
print("total training characters: ", sum([len(x) for x in docs]))

model = train_model(docs = docs, seed = args.seed, train_handle = train_handle, method = args.method, init_method = args.init_method, num_batches = args.num_batches, batch_size = args.batch_size, actions_denom = args.actions_denom, language = args.language, train_ex = args.train_ex, reg_model = args.reg_model, param_method = args.regress_theta)

for test_set in args.test_sets:
    eval_dataset(test_set, data_dir, args.language, model)