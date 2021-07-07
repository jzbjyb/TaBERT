from typing import List, Dict
import json
from pathlib import Path
import random
from tqdm import tqdm
import re
import logging


class WikiSQL(object):
    def __init__(self, root_dir: Path):
        for split in ['train', 'dev', 'test']:
            setattr(self, f'{split}_data', self.load(root_dir / f'{split}.jsonl', root_dir / f'{split}.tables.jsonl'))

    def load(self, example_file: str, table_file: str):
        id2table: Dict[str, Dict] = {}
        with open(table_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                id2table[l['id']] = l
        data: List[Dict] = []
        with open(example_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                l['table'] = id2table[l['table_id']]
                data.append(l)
        return data

    @staticmethod
    def normalize_rows(rows: List[List]):
        return [[str(cell).replace(' ,', ',') for cell in row] for row in rows]

    @staticmethod
    def float_eq(a: str, b: str):
        try:
            a = float(a)
            b = float(b)
            return a == b
        except ValueError:
            return False

    def convert_to_tabert_format(self, split: str, output_path: Path):
        count = num_rows = num_cols = num_used_cols = 0
        data = getattr(self, '{}_data'.format(split))
        all_types = set()
        with open(output_path, 'w') as fout:
            for idx, example in tqdm(enumerate(data)):
                td = {
                    'uuid': None,
                    'table': {'caption': '', 'header': [], 'data': [], 'used_header': []},
                    'context_before': [],
                    'context_after': []
                }
                td['uuid'] = f'wsql_{split}_{idx}'
                question = example['question']
                td['context_before'].append(question)
                td['table']['data'] = self.normalize_rows(example['table']['rows'])
                num_rows += len(td['table']['data'])

                # extract column name
                for cname, ctype in zip(example['table']['header'], example['table']['types']):
                    td['table']['header'].append({
                        'name': '',
                        'name_tokens': None,
                        'type': 'text',
                        'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                        'sample_value_tokens': None,
                        'is_primary_key': False,
                        'foreign_key': None,
                        'used': False,
                        'value_used': False,
                    })
                    td['table']['header'][-1]['name'] = cname
                    td['table']['header'][-1]['type'] = ctype if ctype == 'text' else 'real'
                    all_types.add(ctype)
                num_cols += len(td['table']['header'])

                # extract value, and used
                sql = example['sql']
                num_used_cols += len(set([sql['sel']] + [c[0] for c in sql['conds']]))
                td['table']['header'][sql['sel']]['used'] = True
                for col_ind, _, cond in sql['conds']:
                    td['table']['header'][col_ind]['used'] = True
                    cond = str(cond).replace(' ,', ',')
                    conds = [cond, re.sub(r'\.0$', '', cond), cond.title()]  # match candidates
                    column_data = [row[col_ind] for row in td['table']['data']]
                    value = None
                    for cond in conds:
                        for v in column_data:
                            if cond == v or cond.lower() == v.lower() or self.float_eq(cond, v):
                                value = v
                                break
                        if value is not None:
                            break
                    if value is None:
                        logging.warn(f'{conds} not in data {column_data} for {question}')
                        td['table']['header'][col_ind]['sample_value']['value'] = random.choice(column_data)
                    else:
                        td['table']['header'][col_ind]['sample_value']['value'] = value
                        td['table']['header'][col_ind]['value_used'] = True
                count += 1
                fout.write(json.dumps(td) + '\n')
        print('total count {}, used columns {}/{}'.format(count, num_used_cols, num_cols))