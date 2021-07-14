from typing import List, Dict, Set, Tuple, Any
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import csv
from table_bert.dataset_utils import BasicDataset


class WikiTQ(BasicDataset):
    def __init__(self, root_dir: Path):
        self.train_data = self.load(root_dir / 'random-split-5-train.tsv')
        self.dev_data = self.load(root_dir / 'random-split-5-dev.tsv')
        self.test_data = self.load(root_dir / 'pristine-unseen-tables.tsv')
        self.table_root_dir = root_dir

    @staticmethod
    def get_table(filename: Path):
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',', doublequote=False, escapechar='\\')
            header = [str(c) for c in next(csv_reader)]
            data = [[str(c) for c in row] for row in csv_reader]
            for row in data:
                assert len(row) == len(header), f'{filename} format error for line {row} with #{len(row)}'
            header_types = ['text'] * len(header)
            if len(data) > 0:
                header_types = ['real' if WikiTQ.is_number(cell) else 'text' for cell in data[0]]
            return header, header_types, data

    def load(self, filename: Path):
        numans2count: Dict[int, int] = defaultdict(lambda: 0)
        data: List[Dict] = []
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter='\t')
            _ = next(csv_reader)  # skip tsv head
            for row in csv_reader:
                id, utterance, table_id, targets = row
                targets = targets.split('|')
                numans2count[len(targets)] += 1
                data.append({'id': id, 'utterance': utterance, 'table_id': table_id, 'targets': targets})
        return data

    def convert_to_tabert_format(self, split: str, output_path: Path):
        count = num_rows = num_cols = num_used_cols = 0
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for idx, example in tqdm(enumerate(data)):
                td = {
                    'uuid': None,
                    'table': {'caption': '', 'header': [], 'data': [], 'used_header': []},
                    'context_before': [],
                    'context_after': []
                }
                td['uuid'] = f'wtq_{split}_{idx}'
                question = example['utterance']
                table_header, header_types, table_data = self.get_table(self.table_root_dir / example['table_id'])

                td['context_before'].append(question)
                td['table']['data'] = table_data
                num_rows += len(td['table']['data'])

                # extract column name
                for cname, ctype in zip(table_header, header_types):
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
                num_cols += len(td['table']['header'])

                # TODO: use string matching to find used columns and cells
                count += 1
                fout.write(json.dumps(td) + '\n')
        print('total count {}, used columns {}/{}'.format(count, num_used_cols, num_cols))
