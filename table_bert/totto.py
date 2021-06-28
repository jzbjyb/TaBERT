from typing import List, Dict, Set
import json
from pathlib import Path
from collections import defaultdict
import random


class Totto(object):
    def __init__(self, root_dir: Path):
        self.train_data = self.load(root_dir / 'totto_train_data.jsonl')
        self.dev_daat = self.load(root_dir / 'totto_dev_data.jsonl')

    def load(self, filename: Path):
        data: List[Dict] = []
        with open(filename, 'r') as fin:
            for l in fin:
                data.append(json.loads(l))
        return data

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_cell(cell):
        return cell['row_span'] == 1 and cell['column_span'] == 1

    @staticmethod
    def is_valid_header(cell):
        return cell['is_header'] and Totto.is_valid_cell(cell)

    def convert_to_tabert_format(self, split: str, output_path: Path):
        count = num_rows = num_cols = num_used_rows = num_used_cols = 0
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for example in data:
                td = {
                    'uuid': None,
                    'table': {'caption': '', 'header': [], 'data': [], 'used_header': []},
                    'context_before': [],
                    'context_after': []
                }
                td['uuid'] = 'totto_{}'.format(example['example_id'])
                td['context_before'].append(example['sentence_annotations'][0]['final_sentence'])  # use the first annotated sentence
                row2count: Dict[int, int] = defaultdict(lambda: 0)
                col2rows: Dict[int, Set] = defaultdict(set)
                for r, c in example['highlighted_cells']:
                    row2count[r] += 1
                    col2rows[c].add(r)
                table = example['table']
                num_rows += len(table)
                num_cols += len(table[0]) if len(table) > 0 else 0
                num_used_rows += len(row2count)
                num_used_cols += len(col2rows)
                invalid_table = False
                # extract data
                for row in table[1:]:
                    r = [col['value'] for col in row if not col['is_header'] and Totto.is_valid_cell(col)]
                    if len(r) < len(row):
                        invalid_table = True
                        break
                    td['table']['data'].append(r)
                if invalid_table:
                    continue
                # extract column name
                for col in table[0]:
                    if not Totto.is_valid_header(col):
                        invalid_table = True
                        break
                    td['table']['header'].append({
                        'name': '',
                        'name_tokens': None,
                        'type': 'text',
                        'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                        'sample_value_tokens': None,
                        'is_primary_key': False,
                        'foreign_key': None,
                        'used': False,
                    })
                    td['table']['header'][-1]['name'] = col['value']
                if invalid_table:
                    continue
                # extract column type and value
                for col_idx, col in enumerate(td['table']['header']):
                    if col_idx in col2rows:  # highlight column
                        ''' see 3182391131405542101 for an example
                        if 0 in col2rows:  # highlight column is impossible, which mean the header is fake
                            invalid_table = True
                            break
                        '''
                        row_idx = random.choice(list(col2rows[col_idx]))  # skip the header row
                        col['used'] = True

                    else:
                        row_idx = random.randint(1, len(table) - 1)  # assume the first row is header
                    cell = table[row_idx][col_idx]
                    col['type'] = 'real' if Totto.is_number(cell['value']) else 'text'
                    col['sample_value']['value'] = cell['value']
                if invalid_table:
                    continue
                for row in td['table']['data']:
                    if len(row) != len(td['table']['header']):
                        invalid_table = True
                        break
                if invalid_table:
                    continue
                # add fake header in data if row 0 is highlighted
                if 0 in row2count:
                    td['table']['data'].insert(0, [c['name'] for c in td['table']['header']])
                fout.write(json.dumps(td) + '\n')
                count += 1
        print('total count {}, used rows {}/{}, used columns {}/{}'.format(
            count, num_used_rows, num_rows, num_used_cols, num_cols))
