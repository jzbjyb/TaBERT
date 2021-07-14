from typing import List, Dict, Set, Tuple, Any
import json
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import csv
from table_bert.dataset_utils import BasicDataset


class TableFact(BasicDataset):
    def __init__(self, root_dir: Path):
        self.table_dir = root_dir / 'all_csv'
        self.full_data = self.load(root_dir / 'full_cleaned.json')

    def load(self, example_file: Path):
        with open(example_file, 'r') as fin:
            data = json.load(fin)
            return data

    @staticmethod
    def get_table(filename: str):
        rows = []
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter='#')
            for row in csv_reader:
                rows.append(row)
        header = rows[0]  # assume the first row is header
        data = rows[1:]
        return header, data

    @staticmethod
    def parse_context(context: str):
        raw_sentence: List[str] = []
        mentions: List[Tuple[int, int]] = []
        for i, piece in enumerate(context.split('#')):
            if i % 2 == 1:  # entity
                entity, idx = piece.rsplit(';', 1)
                row_ind, col_ind = idx.split(',')
                row_ind, col_ind = int(row_ind), int(col_ind)
                if row_ind != -1 and col_ind != -1:
                    mentions.append((row_ind, col_ind))
                raw_sentence.append(entity)
            else:  # no entity
                raw_sentence.append(piece)
        return ''.join(raw_sentence), mentions

    def convert_to_tabert_format(self, split: str, output_path: Path):
        count = num_rows = num_cols = num_used_rows = num_used_cols = 0
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for table_id in tqdm(data):
                example = data[table_id]
                caption = example[3]
                # parse table
                header, table_data = self.get_table(self.table_dir / table_id)
                # get types
                header_types = ['real' if self.is_number(cell.lower().strip()) else 'text'
                                for cell in table_data[0]] if len(table_data) > 0 else ['text'] * len(header)
                assert len(header_types) == len(header)
                for context_id, (context, label) in enumerate(zip(example[0], example[1])):
                    if not label:
                        continue
                    context, mentions = self.parse_context(context)
                    num_used_rows += len(set([ri for ri, ci in mentions]) - {0})  # remove header
                    col2rows: Dict[int, Set[int]] = defaultdict(set)
                    for ri, ci in mentions:
                        col2rows[ci].add(ri)
                    td = {
                        'uuid': None,
                        'table': {'caption': caption, 'header': [], 'data': [], 'used_header': []},
                        'context_before': [],
                        'context_after': []
                    }
                    td['uuid'] = f'tablefact_{split}_{table_id}_{context_id}'
                    td['context_before'].append(context)
                    td['table']['data'] = table_data
                    num_rows += len(td['table']['data'])

                    # extract value and used
                    for col_ind, (cname, ctype) in enumerate(zip(header, header_types)):
                        td['table']['header'].append({
                            'name': cname,
                            'name_tokens': None,
                            'type': ctype,
                            'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                            'sample_value_tokens': None,
                            'is_primary_key': False,
                            'foreign_key': None,
                            'used': False,
                            'value_used': False,
                        })
                        td['table']['header'][-1]['used'] = len(col2rows[col_ind]) > 0
                        num_used_cols += int(len(col2rows[col_ind]) > 0)
                        used_rows = list(col2rows[col_ind] - {0})  # remove the header
                        td['table']['header'][-1]['value_used'] = len(used_rows) > 0
                        if len(used_rows) > 0:
                            value = table_data[random.choice(used_rows) - 1][col_ind]  # remove the header
                        else:
                            value = table_data[random.randint(0, len(table_data) - 1)][col_ind]
                        td['table']['header'][-1]['sample_value']['value'] = value
                    num_cols += len(td['table']['header'])

                    count += 1
                    fout.write(json.dumps(td) + '\n')
        print('total count {}, used rows {}/{}, used columns {}/{}'.format(
            count, num_used_rows, num_rows, num_used_cols, num_cols))
