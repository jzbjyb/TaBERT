from argparse import ArgumentParser
import json
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True)
    args = parser.parse_args()

    is_sames = []
    with open(args.prediction, 'r') as fout:
        for i, l in enumerate(fout):
            l = json.loads(l)
            tgt = l['tgt']
            pred = l['pred']
            gold = l['gold']

            assert len(tgt) == len(pred) == len(gold), 'inconsistent number of tokens'
            for tgt_t, pred_t, gold_t in zip(tgt, pred, gold):
                if gold_t is None:
                    continue
                is_sames.append(pred_t == gold_t)

    print(f'average accuracy {np.mean(is_sames)}')
