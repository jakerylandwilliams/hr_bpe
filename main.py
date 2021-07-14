import os
import json

from src.bpe import GreedyBPE
from src.bpe import HRBPE


if __name__ == '__main__':
    os.makedirs('cache/', exist_ok=True)

    seed = 1234

    init_method = 'char'
    # init_method = 'warm'
    # init_method = 'rand'

    num_batches = 1_000
    # num_batches = 10_000

    # batch_size = 1
    # batch_size = 10
    # batch_size = 100
    batch_size = 2500

    # method = 'greedy'
    method = 'hr-bpe'

    # reg_model = 'simon'
    reg_model = 'mixing'
    # reg_model = '?'

    # param_method = 'est_type'
    param_method = 'est_doc'
    # param_method = 'est_theta'
    # param_method = 'regress'
    # param_method = 'regress_theta'

    docs = [x['text'] for x in json.load(open('./data/newstweet-sample-linked.json')) if x['tweets']]

    if method == 'greedy':
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}'
        model = GreedyBPE()
    elif method == 'hr-bpe':
        model_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{reg_model}_{param_method}'
        model = HRBPE(param_method=param_method, reg_model=reg_model)
    else:
        raise ValueError
    try:
        model.load('cache/' + model_str + '.json')
    except FileNotFoundError:
        model.init(docs, seed=seed, method=init_method)
        model.fit(num_batches, batch_size, seed=seed)

        model.save('cache/' + model_str + '.json')

    model.display(model_type=reg_model, method=param_method)

    if method == 'hr-bpe':
        model.display_epochs()

    print(model.encode('Hunter Heidenreich'))