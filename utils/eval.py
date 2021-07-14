from argparse import ArgumentParser
import json
import numpy as np
import random
import csv
from collections import defaultdict
from table_bert.dataset_utils import BasicDataset
from utils.wtq_evaluator import to_value
AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--gold', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--data', type=str, choices=['wikisql', 'wtq'])
    args = parser.parse_args()

    ems = []
    tapas_ems = []  # follow the tapas filtering conditions
    num_cells = []
    agg2ems = defaultdict(list)
    agg2cases = defaultdict(lambda: ([], []))

    cond2ems = defaultdict(list)
    cond2cases = defaultdict(lambda: ([], []))

    numcond2ems = defaultdict(list)
    numcell2ems = defaultdict(list)

    firstword2ems = defaultdict(list)
    firstword2cases = defaultdict(lambda: ([], []))

    numcoord2ems = defaultdict(list)
    answertype2ems = defaultdict(list)

    prev_example = None
    with open(args.prediction, 'r') as pfin, open(args.gold, 'r') as gfin:
        #csv_reader = csv.reader(gfin, delimiter='\t')
        #_ = next(csv_reader)  # skip tsv head
        for i, p in enumerate(pfin):
            p = p.rstrip('\n').split('\t')
            if len(p) == 2:
                pred, gold = p[0].strip(), p[1].strip()
            elif len(p) == 3:
                pred, gold, source = p[0].strip(), p[1].strip(), p[2].strip()
                source.replace('<pad>', '')
                num_cell = len(source.split('</s>')) - 1
                num_cells.append(num_cell)
                first_word = source.split()[1].lower()  # skip cls
            if args.data == 'wikisql':  # exact match
                em = pred.lower() == gold.lower()
            elif args.data == 'wtq':  # official evaluation
                em = to_value(gold).match(to_value(pred))
            else:
                raise NotImplementedError
            ems.append(em)
            anstype = 'number' if BasicDataset.is_number(gold.strip()) else 'text'
            answertype2ems[anstype].append(em)

            example = prev_example = json.loads(gfin.readline())
            #example = next(csv_reader)
            if 'sql' in example:  # wikisql example
                agg = example['sql']['agg']
                num_cond = len(example['sql']['conds'])
                agg2ems[AGG_OPS[agg]].append(em)
                numcond2ems[num_cond].append(em)
                agg2cases[AGG_OPS[agg]][int(em)].append((source, pred, gold))
                for cond in example['sql']['conds']:
                    cond = cond[1]
                    cond2ems[COND_OPS[cond]].append(em)
                    cond2cases[COND_OPS[cond]][int(em)].append((source, pred, gold))
            else:  # tabert format
                num_coord = len(example['answer_coordinates'])
                numcoord2ems[num_coord].append(em)
                if anstype == 'number' or num_coord == 1:
                    tapas_ems.append(em)

            if len(p) == 3:
                numcell2ems[num_cell].append(em)
                firstword2ems[first_word].append(em)
                firstword2cases[first_word][int(em)].append((source, pred, gold))

    print(f'Exact match: [Overall] {np.mean(ems)} [TAPAS] {np.mean(tapas_ems)}, avg #cell {np.mean(num_cells)}')

    firstword2ems = {k: v for k, v in firstword2ems.items() if len(v) >= 10}
    numcell2ems = {k: v for k, v in numcell2ems.items() if len(v) >= 100}
    for k2em, name in [(agg2ems, 'agg'), (cond2ems, 'cond'), (numcond2ems, '#cond'),
                       (numcell2ems, '#cell'), (firstword2ems, 'first word'),
                       (numcoord2ems, '#coord'), (answertype2ems, 'ans type')]:
        print(f'\n===== {name} ======')
        for key, ems in sorted(k2em.items(), key=lambda x: x[0] if type(x[0]) in {float, int} else -len(x[1])):
            print(f'{key}\t\t{np.mean(ems)}\t{len(ems)}')

    if args.output:
        with open(args.output, 'w') as fout:
            for k2em, name in [(agg2cases, 'agg'), (cond2cases, 'cond'), (firstword2cases, 'first word')]:
                fout.write(f'\n===== {name} cases ======\n')
                for key, (badcases, goldcases) in sorted(k2em.items(), key=lambda x: x[0] if type(x[0]) in {float, int} else -len(x[1])):
                    random.shuffle(badcases)
                    for source, pred, gold in badcases[:5]:
                        fout.write(f'{pred}\t{gold}\t{source}\n')
