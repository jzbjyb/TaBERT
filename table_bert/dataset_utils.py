from typing import Dict, List, Tuple
from collections import defaultdict
import json
import csv
import re

class BasicDataset(object):
    @staticmethod
    def is_number(s):
        if s is None:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def float_eq(a: str, b: str):
        try:
            a = float(a)
            b = float(b)
            return a == b
        except ValueError:
            return False

    @staticmethod
    def locate_in_table(values: List[str], table: List[List[str]]):
        locations: List[Tuple[int, int]] = []
        values = set([re.sub('[\W_]+', ' ', v.strip().lower()) for v in values])  # only keep alphanumerics
        for row_idx, row in enumerate(table):
            for col_idx, cell in enumerate(row):
                if re.sub('[\W_]+', ' ', cell.strip().lower()) in values:
                    locations.append((row_idx, col_idx))
        return locations

    @staticmethod
    def add_answer(prep_file: str, answer_file: str, out_file: str, string_match: bool = False):
        numans2count: Dict[int, int] = defaultdict(lambda: 0)
        matched = False
        found_count = total_count = 0
        with open(prep_file, 'r') as pfin, open(answer_file, 'r') as afin, open(out_file, 'w') as fout:
            csv_reader = csv.reader(afin, delimiter='\t')
            _ = next(csv_reader)  # skip tsv head
            for i, l in enumerate(pfin):
                l = json.loads(l)
                al = next(csv_reader)
                coord = eval(al[5])
                anss = eval(al[6])
                assert type(coord) is list and type(anss) is list, 'format error'
                if string_match:
                    l['answer_coordinates'] = BasicDataset.locate_in_table(anss, l['table']['data'])
                    found_count += int(len(l['answer_coordinates']) > 0)
                    total_count += 1
                else:
                    l['answer_coordinates'] = [eval(c) for c in coord]
                l['answers'] = anss
                numans2count[len(anss)] += 1
                fout.write(json.dumps(l) + '\n')
            try:
                next(csv_reader)
            except StopIteration as e:
                matched = True
            if not matched:
                raise Exception(f'{prep_file} and {answer_file} have different #rows')
        print('#ans -> count: {}'.format(sorted(numans2count.items())))
        print(f'found {found_count} out of {total_count}')
