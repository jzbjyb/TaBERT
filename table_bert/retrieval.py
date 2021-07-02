from typing import List, Tuple, Set, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import string
from typing import Iterator
import json
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import subprocess
import copy
import random
import numpy as np


class ESWrapper():
    MAX_QUERY_LEN = 4096
    PUNCT_TO_SPACE = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def __init__(self, index_name: str):
        self.es = Elasticsearch()
        self.index_name = index_name

    def lucene_format(self, query_str: str, field: str):
        new_query_str = query_str.translate(self.PUNCT_TO_SPACE)
        new_query_str = new_query_str.replace(' AND ', ' ').replace(' and ', ' ')
        q = '{}:({})'.format(field, new_query_str[:self.MAX_QUERY_LEN])
        return q

    def get_topk(self, query_str: str, field: str, topk: int = 5):
        if len(query_str) <= 0:
            return []
        results = self.es.search(
            index=self.index_name,
            body={'query': {'match': {field: query_str[:self.MAX_QUERY_LEN]}}},
            size=topk)['hits']['hits']
        return [(doc['_source'], doc['_score']) for doc in results]

    def build_index(self, doc_iter: Iterator):
        print('delete index')
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        print('create index')
        print(self.es.indices.create(index=self.index_name, ignore=400))
        print('add docs')
        print(bulk(self.es, doc_iter))

    @staticmethod
    def format_text(example):
        return ' '.join(example['context_before'] + example['context_after'])

    @staticmethod
    def format_table(example):
        return ' & '.join(['{} | {} | {}'.format(
            h['name'], h['type'], h['sample_value']['value']) for h in example['table']['header']])

    def table_text_data_iterator(self, filename: str, split: str = 'train', max_length: int = 10000):
        with open(filename, 'r') as fin:
            for idx, l in tqdm(enumerate(fin)):
                example = json.loads(l)
                text = self.format_text(example)[:max_length]
                table = self.format_table(example)[:max_length]
                yield {
                    '_index': self.index_name,
                    '_type': 'document',
                    'id': example['uuid'],
                    'line_idx': idx,
                    'split': split,
                    'text': text,
                    'table': table
                }


def retrieve_part(filename, topk, two_idx):  # inclusive and exclusive
    start_idx, end_idx = two_idx
    results: List[Tuple[int, List, List]] = []
    with open(filename, 'r') as fin:
        for idx, l in tqdm(enumerate(fin), total=end_idx - start_idx, disable=start_idx != 0):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
            example = json.loads(l)
            text, table = ESWrapper.format_text(example), ESWrapper.format_table(example)
            bytext = [(doc['line_idx'], score) for doc, score in es.get_topk(text, field='text', topk=topk + 1)]
            bytable = [(doc['line_idx'], score) for doc, score in es.get_topk(table, field='table', topk=topk + 1)]
            results.append((idx, bytext, bytable))
    return results


def retrieve(filename: str, output: str, topk: int = 5, threads: int = 1):
    total_count = int(subprocess.check_output('wc -l {}'.format(filename), shell=True).split()[0])
    bs = total_count // threads
    splits = [(i * bs, (i * bs + bs) if i < threads - 1 else total_count) for i in range(threads)]
    print('splits:', splits)

    pool = mp.Pool(threads)
    results_list = pool.map(partial(retrieve_part, filename, topk), splits)

    format_list = lambda l: ' '.join(['{},{}'.format(i, s) for i, s in l])
    with open(output, 'w') as fout:
        for results in results_list:
            for idx, bytext, bytable in results:
                fout.write('{}\t{}\t{}\n'.format(idx, format_list(bytext), format_list(bytable)))


def fill_to_at_middle(lst: List, fill_to: int):
    if len(lst) == 0:
        raise ValueError()
    if len(lst) > fill_to:
        lst = lst[:fill_to]
    if len(lst) == fill_to:
        return lst
    mid = min(len(lst) // 2 + len(lst) % 2, len(lst))
    lst = lst[:mid] + ([lst[mid - 1]] * (fill_to - len(lst))) + lst[mid:]
    return lst


def load_neg_file(neg_file: str, fill_to: int = None):
    idx2neg: Dict[int, Tuple[List, List]] = {}
    with open(neg_file, 'r') as fin:
        for l in fin:
            idx, bytext, bytable = l.strip().split('\t')
            idx = int(idx)
            # only get idx
            bytext = [int(s.split(',')[0]) for s in bytext.split(' ')]
            bytable = [int(s.split(',')[0]) for s in bytable.split(' ')]
            bytext = [i for i in bytext if i != idx]
            bytable = [i for i in bytable if i != idx]
            if fill_to:
                bytext = fill_to_at_middle(bytext, fill_to=fill_to)
                bytable = fill_to_at_middle(bytable, fill_to=fill_to)
            idx2neg[idx] = (bytext, bytable)
    return idx2neg


def get_from_merge(lst: List[Tuple], count: int, from_bottom: bool = False, filter_set: Set = set()):
    cur = len(lst) - 1 if from_bottom else 0
    off = -1 if from_bottom else 1
    result = set()
    while 0 <= cur < len(lst) and len(result) < count:
        for i in lst[cur]:
            if i in filter_set:
                continue
            result.add(i)
            if len(result) >= count:
                break
        cur += off
    assert len(result) == count
    return list(result)


def combine_negative(data_file: str, neg_file: str, output: str, fill_to: int, num_top_neg: int, num_bottom_neg: int,
                     num_random_neg: int):
    idx2neg = load_neg_file(neg_file, fill_to=fill_to)
    idx2example: Dict[int, Dict] = {}
    with open(data_file, 'r') as fin:
        for idx, l in enumerate(fin):
            idx2example[idx] = json.loads(l)
            idx2example[idx]['is_positive'] = True
    total_num = len(idx2example)
    with open(output, 'w') as fout:
        for idx in tqdm(range(total_num)):
            # pos
            fout.write(json.dumps(idx2example[idx]) + '\n')

            # neg
            bytext, bytable = idx2neg[idx]
            merge = list(zip(bytext, bytable))

            negs = []
            negs += get_from_merge(merge, count=num_top_neg)
            negs += get_from_merge(merge, count=num_bottom_neg, from_bottom=True, filter_set=set(negs))
            rand_negs = np.random.choice(total_num, num_random_neg + len(negs) + 1, replace=False)  # avoid collision
            sn = set(negs)
            rand_negs = [i for i in rand_negs if i != idx and i not in sn][:num_random_neg]
            negs += rand_negs
            assert len(negs) == len(set(negs)), 'different neg methods have collision'
            assert len(negs) == (num_top_neg + num_bottom_neg + num_random_neg), '#neg not enough'

            neg_example = copy.deepcopy(idx2example[idx])
            neg_example['is_positive'] = False
            for i in negs:
                neg_example['uuid'] = idx2example[idx]['uuid'] + '_{}-{}'.format(idx, i)
                neg_example['table'] = idx2example[i]['table']  # replace table
                fout.write(json.dumps(neg_example) + '\n')


if __name__ == '__main__':
    # index and search
    filename = 'data/totto_data/train/preprocessed.jsonl'
    neg_file = 'test.tsv'
    topk = 100
    es = ESWrapper('totto')
    es.build_index(es.table_text_data_iterator(filename))
    retrieve(filename, neg_file, topk=topk, threads=10)

    # get negative
    final_file = 'test.jsonl'
    combine_negative(filename, neg_file, final_file, topk, 3, 3, 3)
