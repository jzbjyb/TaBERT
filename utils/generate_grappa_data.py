import os
os.environ['WANDB_API_KEY'] = '9caada2c257feff1b6e6a519ad378be3994bc06a'

from typing import List, Dict, Tuple, Set
import functools
import time
from argparse import ArgumentParser
from pathlib import Path
import os
import random
import numpy as np
import json
import copy
from collections import defaultdict
from tqdm import tqdm
import multiprocessing
from table_bert.totto import Totto
from table_bert.wikisql import WikiSQL
from table_bert.tablefact import TableFact
from table_bert.wikitablequestions import WikiTQ
from table_bert.turl import TurlData
from table_bert.tapas import TapasTables
from table_bert.dataset_utils import BasicDataset
from table_bert.dataset import Example


def tableshuffle(prep_file: str, output_file: str):
    with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
        for l in tqdm(fin):
            example = json.loads(l)
            Example.shuffle_table(example)
            fout.write(json.dumps(example) + '\n')


def find_other_table(prep_file: str, output_file: str, max_count: int):
    tablecell2ind: Dict[str, List] = defaultdict(list)
    examples: List[Dict] = []
    with open(prep_file, 'r') as fin:
        for eid, l in tqdm(enumerate(fin), desc='build tablecell2ind'):
            example = json.loads(l)
            examples.append(example)
            for row in example['table']['data']:
                for cell in row:
                    cell = cell.strip().lower()
                    if len(tablecell2ind[cell]) > 0 and tablecell2ind[cell][-1] == eid:
                        continue
                    tablecell2ind[cell].append(eid)
    match_counts = []
    used_eids: Set[int] = set()
    with open(output_file, 'w') as fout:
        for eid, example in tqdm(enumerate(examples), desc='find tables'):
            context = example['context_before'][0]
            eid2mentions: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
            all_mentions: Set[Tuple[int, int]] = set()
            for s, e in example['context_before_mentions'][0]:
                all_mentions.add((s, e))
                kw = context[s:e].strip().lower()
                for _eid in random.sample(tablecell2ind[kw], min(len(tablecell2ind[kw]), 1000)):
                    if _eid == eid:
                        continue
                    eid2mentions[_eid].add((s, e))
            if len(eid2mentions) > 0:
                match_eid, mentions = max(eid2mentions.items(), key=lambda x: len(x[1]))
            else:
                match_eid, mentions = (random.randint(0, len(examples) - 1), set())
            used_eids.add(match_eid)
            ne = copy.deepcopy(example)
            ne['table'] = examples[match_eid]['table']
            ne['context_before_mentions'] = [list(mentions) + list(all_mentions - mentions)]
            assert len(ne['context_before_mentions'][0]) == len(all_mentions), f"{ne['context_before_mentions'][0]} {all_mentions}"
            fout.write(f'{json.dumps(ne)}\n')
            match_counts.append(min(max_count, len(mentions)))
        print(np.mean(match_counts))
        print(f'#used_eids {len(used_eids)}')


def _generate_retrieval_data_single(example_lines: List[str], ret_examples_li: List[List[Dict]], bywhich: str, max_context_len: int, max_num_rows: int):
    examples = []
    for example_line, ret_examples in zip(example_lines, ret_examples_li):
        example = json.loads(example_line)
        best_match_mentions = None
        best_match = None
        for _example in ret_examples:
            if bywhich == 'context':
                context = example['context_before'][0]
                table = _example['table']['data']
            elif bywhich == 'table':
                context = _example['context_before'][0]
                table = example['table']['data']
            else:
                raise NotImplementedError
            if max_context_len:
                context = context[:max_context_len]
            if max_num_rows:
                table = table[:max_num_rows]
            locations = BasicDataset.get_mention_locations(context, table)
            if best_match_mentions is None or len(locations) > len(best_match_mentions):
                best_match_mentions = locations
                best_match = _example
        if best_match is not None:
            if bywhich == 'context':
                example['table'] = best_match['table']
                example['context_before_mentions'] = [best_match_mentions]
            elif bywhich == 'table':
                example['context_before'] = best_match['context_before']
                example['context_before_mentions'] = [best_match_mentions]
        examples.append(example)
    return examples


def generate_retrieval_data(retrieval_file: str, target_file: str, source_file: str, output_file: str,
                            bywhich: str, topk: int, nthread: int, batch_size: int = 100,
                            max_context_len: int = None, max_num_rows: int = None,
                            remove_self: bool = False, only_self: bool = False):
    assert bywhich in {'context', 'table'}
    idx2example: Dict[int, Dict] = {}
    with open(source_file, 'r') as fin:
        for idx, l in enumerate(fin):
            idx2example[idx] = json.loads(l)
    pool = multiprocessing.Pool(processes=nthread)
    start = time.time()
    with open(retrieval_file, 'r') as fin, open(target_file, 'r') as tfin, open(output_file, 'w') as fout:
        example_lines = []
        ret_examples_li = []
        results = []
        tfin_idx = -1
        def get_next_target_until(idx):
            nonlocal tfin_idx
            l = None
            while tfin_idx < idx:
                l = tfin.readline()
                tfin_idx += 1
            return l
        for l in tqdm(fin, miniters=50):
            idx, bytext, bytable = l.rstrip('\n').split('\t')
            idx = int(idx)
            if only_self:
                byall = [idx]
            else:
                bytext = [int(s.split(',')[0]) for s in bytext.split(' ') if len(s) > 0][:topk]
                bytable = [int(s.split(',')[0]) for s in bytable.split(' ') if len(s) > 0][:topk]
                byall = list(set(bytext + bytable) - ({idx} if remove_self else set()))[:topk]
            ret_examples = [idx2example[_idx] for _idx in byall]
            example_line = get_next_target_until(idx)
            ret_examples_li.append(ret_examples)
            example_lines.append(example_line)
            if len(example_lines) >= batch_size:
                r = pool.apply_async(
                    functools.partial(_generate_retrieval_data_single,
                                      bywhich=bywhich, max_context_len=max_context_len, max_num_rows=max_num_rows),
                    (example_lines, ret_examples_li))
                results.append(r)
                example_lines = []
                ret_examples_li = []
                if len(results) == nthread:
                    for r in results:
                        for e in r.get():
                            fout.write(json.dumps(e) + '\n')
                    results = []
        if len(example_lines) >= 0:
            r = pool.apply_async(
                functools.partial(_generate_retrieval_data_single,
                                  bywhich=bywhich, max_context_len=max_context_len, max_num_rows=max_num_rows),
                (example_lines, ret_examples_li))
            results.append(r)
        if len(results) > 0:
            for r in results:
                for e in r.get():
                    fout.write(json.dumps(e) + '\n')
    end = time.time()
    print(f'total time {end - start}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=[
        'totto', 'wikisql', 'tablefact', 'wtq', 'turl', 'tapas', 'overlap', 'fakepair', 'retpair', 'tableshuffle'])
    parser.add_argument('--path', type=Path, required=True, nargs='+')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    random.seed(2021)
    np.random.seed(2021)

    # dataset-specific prep
    if args.data == 'totto':
        totto = Totto(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        totto.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention.jsonl')
    elif args.data == 'wikisql':
        wsql = WikiSQL(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        wsql.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention_with_sql.jsonl', add_sql=True)
        WikiSQL.add_answer(args.output_dir / args.split / 'preprocessed_mention_with_sql.jsonl',
                           args.output_dir / 'converted' / f'{args.split}.tsv',  # generated by TAPAS
                           args.output_dir / args.split / 'preprocessed_mention_with_sql_ans.jsonl')
    elif args.data == 'tablefact':
        tf = TableFact(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        tf.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention.jsonl')
    elif args.data == 'wikitq':
        wtq = WikiTQ(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        wtq.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')
        split2file = {
            'train': 'random-split-5-train.tsv',
            'dev': 'random-split-5-dev.tsv',
            'test': 'test.tsv'
        }
        WikiTQ.add_answer(args.output_dir / args.split / 'preprocessed.jsonl',
                          args.output_dir / 'converted' / split2file[args.split],
                          args.output_dir / args.split / 'preprocessed_with_ans.jsonl',
                          string_match=True)
    elif args.data == 'turl':
        avoid_titles = set()
        with open(str(args.path[0] / 'titles_in_3merge.txt'), 'r') as fin:
            for l in fin:
                avoid_titles.add(l.strip())
        turl = TurlData(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        # only use avoid_titles for the test split
        turl.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_cf_avoid3merge.jsonl',
                                      task='cell_filling', avoid_titles=avoid_titles)
        turl.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_sa_avoid3merge.jsonl',
                                      task='schema_augmentation', avoid_titles=avoid_titles)
    elif args.data == 'tapas':
        tt = TapasTables(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        tt.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')

    # others
    elif args.data == 'fakepair':
        find_other_table(args.path[0], args.output_dir, max_count=3)
    elif args.data == 'retpair':
        only_self = False
        remove_self = False
        batch_size = 5000 if only_self else 1000
        retrieval_file, target_file, source_file = args.path
        generate_retrieval_data(retrieval_file, target_file, source_file, args.output_dir,
                                bywhich=args.split, topk=10, nthread=40, batch_size=batch_size,
                                max_context_len=256, max_num_rows=100,  # used for tapas setting
                                remove_self=remove_self, only_self=only_self)
    elif args.data == 'tableshuffle':
        tableshuffle(args.path[0], args.output_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
