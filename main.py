import os
import json

from src.bpe import GreedyBPE
from src.bpe import HRBPE


if __name__ == '__main__':
    os.makedirs('cache/', exist_ok=True)
    os.makedirs('img/', exist_ok=True)
    seed = 691

    # language = 'EN'
#     language = 'PL'
#     language = 'IT'
#     language = 'HI'
#     language = 'HE'
#     language = 'GA'
#     language = 'FR'
#     language = 'EU'
#     language = 'EL'
#     language = 'DE'
    language = 'ZH'
#     language = 'TR'
#     language = 'SV'
#     language = 'RO'
#     language = 'PT'

    init_method = 'char'
    # init_method = 'warm'
    # init_method = 'rand'
    num_batches = 100
    # num_batches = 1_000

    # batch_size = 1
    # batch_size = 10
    # batch_size = 100
    batch_size = 1_000
    # batch_size = 1_000_000
    
    actions_per_batch = int(batch_size/1)

    # method = 'greedy'
    method = 'hr-bpe'
 
    # reg_model = 'simon'
    reg_model = 'mixing'
    # reg_model = 'resonator'

    # param_method = 'est_type'
    # param_method = 'est_doc'
    # param_method = 'est_theta'
    # param_method = 'regress'
    param_method = 'regress_theta'

    early_stop = True
    
    if language == "ZH":
        train_set = 'nlpcc2016-train' # 'icwb2-train-as' # 'icwb2-train-msr' # 'icwb2-train-cityu' # 'icwb2-train-pku' # 
        test_set = 'nlpcc2016-test' # 'icwb2-test-as' # 'icwb2-test-msr' # 'icwb2-test-cityu' # 'icwb2-test-pku' # 
        train_handle = train_set
        docs = ["".join(ts) for ts in json.load(open('./data/gold/'+language+'/' + train_set + '.json')).values()]
        # uncomment this to boost with external data, here as line-by-line raw text
        docs += [x.strip() for x in open('data/nlpcc2016/nlpcc2016-wordseg-background-10k.txt').read().split("\n") if x.strip()]
    elif language == "EN":
        train_sets = ['ewtb', 'lowlands', 'ritter', 'parseme-train']
        test_sets = ['tweebank', 'trustpilot', 'ted', 'parseme-test']
        train_handle = '-'.join(train_sets)
        docs = ["".join(ts) for train_set in train_sets for ts in json.load(open('./data/gold/'+language+'/' + train_set + '.json')).values()]
        # uncomment this to boost with external data, here as a list newstweet sample
        docs += [x['text'] for x in json.load(open('./data/newstweet-sample-linked.json')) if x['tweets']][:500]
    else:
        train_set = 'parseme-train'
        test_set = 'parseme-test' 
        train_handle = train_set
        docs = ["".join(ts) for ts in json.load(open('./data/gold/'+language+'/' + train_set + '.json')).values()]
    
    print('total training characters: ', sum([len(x) for x in docs]))

    if method == 'greedy':
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{language}-{train_handle}'
        model = GreedyBPE()
    elif method == 'hr-bpe':
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{reg_model}_{param_method}_{language}-{train_handle}'
        model = HRBPE(param_method=param_method, reg_model=reg_model, early_stop=early_stop)
    else:
        raise ValueError
    try:
        model.load('cache/' + model_str + '.json')
    except FileNotFoundError:
        model.init(docs, seed=seed, method=init_method)
        model.fit(num_batches, batch_size, actions_per_batch=actions_per_batch, seed=seed)

        model.save('cache/' + model_str + '.json')

    model.display(model_type=reg_model, method=param_method, fname= 'img/' + model_str + '.png')

    if method == 'hr-bpe':
        model.display_epochs(fname = 'img/' + model_str + '.png')

    import numpy as np
    from tqdm import tqdm
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

    def eval_dataset(dataset):
        Ps, Rs, F1s = [], [], []
        for rix, ts in enumerate(tqdm(list(dataset.values()))):
            if not ts:
                continue
            ts_hat = model.tokenize(''.join(ts))
            P, R, F1, = eval_segmentation(ts, ts_hat)
            Ps.append(P); Rs.append(R); F1s.append(F1)
        return Ps, Rs, F1s

    if language == "ZH":
        dataset = json.load(open('./data/gold/'+language+'/' + test_set + '.json'))
        Ps, Rs, F1s = eval_dataset(dataset)
        print(test_set, ': P: ', np.mean(Ps), 'R: ',  np.mean(Rs), 'F1: ', np.mean(F1s))
    elif language == "EN":
        for test_set in test_sets:
            dataset = json.load(open('./data/gold/'+language+'/' + test_set + '.json'))
            Ps, Rs, F1s = eval_dataset(dataset)
            print(test_set, ': P: ', np.mean(Ps), 'R: ',  np.mean(Rs), 'F1: ', np.mean(F1s))
    else:
        dataset = json.load(open('./data/gold/'+language+'/' + test_set + '.json'))
        Ps, Rs, F1s = eval_dataset(dataset)
        print(test_set, ': P: ', np.mean(Ps), 'R: ',  np.mean(Rs), 'F1: ', np.mean(F1s))

