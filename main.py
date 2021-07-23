import os
import json

from src.bpe import GreedyBPE
from src.bpe import HRBPE


if __name__ == '__main__':
    os.makedirs('cache/', exist_ok=True)
    os.makedirs('img/', exist_ok=True)
    seed = 1234

    language = 'EN'
#     language = 'PL'
#     language = 'IT'
#     language = 'HI'
#     language = 'HE'
#     language = 'GA'
#     language = 'FR'
#     language = 'EU'
#     language = 'EL'
#     language = 'DE'
#     language = 'ZH'
#     language = 'TR'
#     language = 'SV'
#     language = 'RO'
#     language = 'PT'

    init_method = 'char'
    # init_method = 'warm'
    # init_method = 'rand'

    num_batches = 100
    # num_batches = 10_000

    # batch_size = 1
    # batch_size = 10
    # batch_size = 100
    batch_size = 1_000_000

    actions_per_batch = batch_size

    # method = 'greedy'
    method = 'hr-bpe'
 
    # reg_model = 'simon'
    reg_model = 'mixing'
    # reg_model = 'resonator'

    param_method = 'est_type'
    # param_method = 'est_doc'
    # param_method = 'est_theta'
    # param_method = 'regress'
    # param_method = 'regress_theta'

    early_stop = False
    
    docs = []
    datasets = []
    filepaths = []
    for filepath in [fpath for fpath in os.listdir('./data/gold/'+language+'/') if '.json' in fpath]:
        datasets.append(json.load(open('./data/gold/'+language+'/'+filepath)))
        filepaths.append(filepath)
        # docs += ["".join(x) for x in datasets[-1].values()]
    
    docs = [x['text'] for x in json.load(open('./data/newstweet-sample-linked.json')) if x['tweets']]

    if method == 'greedy':
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}'
        model = GreedyBPE()
    elif method == 'hr-bpe':
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{reg_model}_{param_method}'
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

#     print(model.encode('Hunter Heidenreich'), [model.decode([x]) for x in model.encode('Hunter Heidenreich')],
#           [model.decode([x]) for x in model.encode('This that and the other thing!')])

    import numpy as np
    def get_locs(tokens):
        return [0] + list(np.cumsum([len(t) for t in tokens])) # list(zip(locs[0:-1],locs[1:]))

    def eval_segmentation(ts, ts_hat):
        y = set(get_locs(ts)); y_hat = set(get_locs(ts_hat))
        TP = len(y_hat.intersection(y)); FP = len(y_hat - y); FN = len(y - y_hat)
        P = TP/(TP+FP) if (TP+FP) else 0
        R = TP/(TP+FN) if (TP+FN) else 0
        F1 = 2*P*R/(P+R) if (P+R) else 0
        return P, R, F1

    def eval_dataset(dataset):
        Ps, Rs, F1s = [], [], []
        for ts in list(dataset.values()):
            if not ts:
                continue
            ts_hat = model.tokenize(''.join(ts))
            print('record: ',ts)
            print('tokens: ',ts_hat)
            if not len("".join(ts)) == len("".join(ts_hat)):
                print('guess:  ',''.join(ts_hat))
            # assert len("".join(ts)) == len("".join(ts_hat))
            P, R, F1, = eval_segmentation(ts, ts_hat)
            Ps.append(P); Rs.append(R); F1s.append(F1)
            print(P, R, F1)
            print("")
        return Ps, Rs, F1s
    test_text = "New York (CNN Business) McDonald's is in damage control mode after a"
    guess_ts = model.tokenize(test_text)
    test_ts = ["New York", " ", "(", "CNN Business", ")", " ", "McDonald's", " ", "is", " ", "in", " ", "damage control", " ", "mode", " ", "after", " ", "a"]
    print("record: ", test_ts)
    print("tokens: ", guess_ts)
    print(eval_segmentation(test_ts, guess_ts))
    print("")
    for di, dataset in enumerate(datasets):
        Ps, Rs, F1s = eval_dataset(dataset)
        print(filepaths[di], np.mean(Ps), np.mean(Rs), np.mean(F1s))
        break

