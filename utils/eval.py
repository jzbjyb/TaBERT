import os
os.environ['USE_TRANSFORMER'] = 'True'  # use new version

from typing import List, Union
from argparse import ArgumentParser
import json
import numpy as np
import random
from collections import defaultdict
import re
from table_bert.dataset_utils import BasicDataset
from table_bert.config import TableBertConfig
from utils.wtq_evaluator import to_value, to_value_list, check_denotation
AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']


def compute_f1(preds: List[str], golds: List[str]):
    sames = set(preds) & set(golds)
    p = len(sames) / (len(preds) or 1)
    r = len(sames) / (len(golds) or 1)
    f1 = 2 * p * r / ((p + r) or 1)
    return f1


def source_contains(source: str, targets: Union[str, List[str]]):
    if type(targets) is str:  targets = [targets]
    source = source.lower()
    for target in targets:
        if target.lower() not in source:
            return False
    return True


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--gold', type=str, required=True)
    parser.add_argument('--multi_ans_sep', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--data', type=str, default='wtq', choices=['wikisql', 'wtq', 'wikisql_sql', 'turl'])
    parser.add_argument('--model_type', type=str, default='facebook/bart-base')
    args = parser.parse_args()
    if '_sql' in args.data:
        from rouge import Rouge
        rouge = Rouge()

    cls_token, sep_token, pad_token = TableBertConfig.get_special_tokens(args.model_type)

    ans_in_inputs = []
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
    with open(args.prediction, 'r', encoding='utf-8') as pfin, \
      open(args.gold, 'r', encoding='utf-8') as gfin:
        #csv_reader = csv.reader(gfin, delimiter='\t')
        #_ = next(csv_reader)  # skip tsv head
        for i, p in enumerate(pfin):
            p = p.rstrip('\n').split('\t')
            if len(p) == 2:
                pred, gold = p[0].strip(), p[1].strip()
            elif len(p) == 3:
                pred, gold, source = p[0].strip(), p[1].strip(), p[2].strip()
                pred = pred.replace(pad_token, '')
                gold = gold.replace(pad_token, '')
                source = source.replace(pad_token, '')
                num_cell = len(source.split(sep_token)) - 1
                num_cells.append(num_cell)
                first_word = source.split()[1].lower()  # skip cls
            if args.data == 'wikisql':  # exact match
                pred = pred.replace(cls_token, '').replace(sep_token, '').strip()
                gold = gold.replace(cls_token, '').replace(sep_token, '').strip()
                em = pred.lower() == gold.lower()
            elif args.data == 'wtq':  # official evaluation
                pred = pred.replace(cls_token, '').replace(sep_token, '').strip()
                gold = gold.replace(cls_token, '').replace(sep_token, '').strip()
                if args.multi_ans_sep:
                    sep = args.multi_ans_sep
                    golds = gold.split(sep)
                    em = check_denotation(to_value_list(golds), to_value_list(pred.split(sep)))
                    ans_in_input = source_contains(source, golds)
                else:
                    em = to_value(gold).match(to_value(pred))
                    ans_in_input = source_contains(source, gold)
                ans_in_inputs.append(ans_in_input)
            elif args.data == 'wikisql_sql':
                em = rouge.get_scores([pred.lower()], [gold.lower()], avg=True)['rouge-l']['f']
            elif args.data == 'turl':
                pred = pred.replace(cls_token, '').replace(sep_token, '')
                gold = gold.replace(cls_token, '').replace(sep_token, '')
                pred = re.sub('\s+', '', pred)
                gold = re.sub('\s+', '', gold)
                preds = [i for i in pred.split('<|>') if i != '']
                golds = [i for i in gold.split('<|>') if i != '']
                #print(preds)
                #print(golds)
                #input()
                em = compute_f1(preds, golds)
            else:
                raise NotImplementedError
            ems.append(em)
            anstype = 'number' if BasicDataset.is_number(gold.strip()) else 'text'
            answertype2ems[anstype].append(em)

            example = prev_example = json.loads(gfin.readline())
            #example = next(csv_reader)
            if 'sql' in example and type(example['sql']) is dict and 'agg' in example['sql']:  # wikisql example
                agg = example['sql']['agg']
                num_cond = len(example['sql']['conds'])
                agg2ems[AGG_OPS[agg]].append(em)
                numcond2ems[num_cond].append(em)
                agg2cases[AGG_OPS[agg]][int(em)].append((source, pred, gold))
                for cond in example['sql']['conds']:
                    cond = cond[1]
                    cond2ems[COND_OPS[cond]].append(em)
                    cond2cases[COND_OPS[cond]][int(em)].append((source, pred, gold))
            elif 'answer_coordinates' in example:  # tabert format
                num_coord = len(example['answer_coordinates'])
                numcoord2ems[num_coord].append(em)
                if anstype == 'number' or num_coord == 1:
                    tapas_ems.append(em)

            if len(p) == 3:
                numcell2ems[num_cell].append(em)
                firstword2ems[first_word].append(em)
                firstword2cases[first_word][int(em)].append((source, pred, gold))

    print(f'Exact match: [Overall] {np.mean(ems)} [TAPAS] {np.mean(tapas_ems)}, avg #cell {np.mean(num_cells)}')
    print(f'Answer in input: {np.mean(ans_in_inputs)}')

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
