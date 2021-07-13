from argparse import ArgumentParser
import json
import numpy as np
import random
import csv
from collections import defaultdict
#from table_bert.wikisql import WikiSQL
AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--gold', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    ems = []
    num_cells = []
    agg2ems = defaultdict(list)
    agg2cases = defaultdict(lambda: ([], []))

    cond2ems = defaultdict(list)
    cond2cases = defaultdict(lambda: ([], []))

    numcond2ems = defaultdict(list)
    numcell2ems = defaultdict(list)

    firstword2ems = defaultdict(list)
    firstword2cases = defaultdict(lambda: ([], []))

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
            example = json.loads(gfin.readline())
            #example = next(csv_reader)
            agg = example['sql']['agg']
            num_cond = len(example['sql']['conds'])
            em = pred.lower() == gold.lower()
            ems.append(em)
            agg2ems[AGG_OPS[agg]].append(em)
            if em:
                agg2cases[AGG_OPS[agg]][0].append((source, pred, gold))
            else:
                agg2cases[AGG_OPS[agg]][1].append((source, pred, gold))
            numcond2ems[num_cond].append(em)
            for cond in example['sql']['conds']:
                cond = cond[1]
                cond2ems[COND_OPS[cond]].append(em)
                if em:
                    cond2cases[COND_OPS[cond]][0].append((source, pred, gold))
                else:
                    cond2cases[COND_OPS[cond]][1].append((source, pred, gold))
            if len(p) == 3:
                numcell2ems[num_cell].append(em)
                firstword2ems[first_word].append(em)
                if em:
                    firstword2cases[first_word][0].append((source, pred, gold))
                else:
                    firstword2cases[first_word][1].append((source, pred, gold))
            #print(p, '#', g)
            #input()

    print(f'Exact match: {np.mean(ems)}, avg #cell {np.mean(num_cells)}')
    firstword2ems = {k: v for k, v in firstword2ems.items() if len(v) >= 10}
    numcell2ems = {k: v for k, v in numcell2ems.items() if len(v) >= 100}
    for k2em, name in [(agg2ems, 'agg'), (cond2ems, 'cond'), (numcond2ems, '#cond'), (numcell2ems, '#cell'), (firstword2ems, 'first word')]:
        print(f'\n===== {name} ======')
        for key, ems in sorted(k2em.items(), key=lambda x: x[0] if type(x[0]) in {float, int} else -len(x[1])):
            print(f'{key}\t\t{np.mean(ems)}\t{len(ems)}')
    if args.output:
        with open(args.output, 'w') as fout:
            for k2em, name in [(agg2cases, 'agg'), (cond2cases, 'cond'), (firstword2cases, 'first word')]:
                fout.write(f'\n===== {name} cases ======\n')
                for key, (goodcases, badcases) in sorted(k2em.items(), key=lambda x: x[0] if type(x[0]) in {float, int} else -len(x[1])):
                    random.shuffle(badcases)
                    for source, pred, gold in badcases[:5]:
                        fout.write(f'{pred}\t{gold}\t{source}\n')
