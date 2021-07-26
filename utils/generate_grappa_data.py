from typing import List, Dict, Tuple, Set
from argparse import ArgumentParser
from pathlib import Path
import os
import random
import numpy as np
import json
import copy
from collections import defaultdict
from tqdm import tqdm
from table_bert.totto import Totto
from table_bert.wikisql import WikiSQL
from table_bert.tablefact import TableFact
from table_bert.wikitablequestions import WikiTQ
from table_bert.turl import TurlData


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
                for _eid in tablecell2ind[kw]:
                    if _eid == eid:
                        continue
                    eid2mentions[_eid].add((s, e))
            eid2mentions = sorted(eid2mentions.items(), key=lambda x: -len(x[1])) or [(random.randint(0, len(examples) - 1), set())]
            eid2mentions = [(e, c) for e, c in eid2mentions if len(c) >= 3] or eid2mentions[:1]
            random.shuffle(eid2mentions)
            match_eid, mentions = eid2mentions[0]
            used_eids.add(match_eid)
            ne = copy.deepcopy(example)
            ne['table'] = examples[match_eid]['table']
            ne['context_before_mentions'] = [list(mentions) + list(all_mentions - mentions)]
            assert len(ne['context_before_mentions'][0]) == len(
                all_mentions), f"{ne['context_before_mentions'][0]} {all_mentions}"
            fout.write(f'{json.dumps(ne)}\n')
            match_counts.append(min(max_count, len(mentions)))
            print(np.mean(match_counts))
        print(f'#used_eids {len(used_eids)}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['totto', 'wikisql', 'tablefact', 'wtq', 'turl', 'overlap', 'fakepair'])
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    random.seed(2021)
    np.random.seed(2021)

    if args.data == 'totto':
        totto = Totto(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        totto.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention.jsonl')
    elif args.data == 'wikisql':
        wsql = WikiSQL(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        wsql.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention_with_sql.jsonl', add_sql=True)
        WikiSQL.add_answer(args.output_dir / args.split / 'preprocessed_mention_with_sql.jsonl',
                           args.output_dir / 'converted' / f'{args.split}.tsv',  # generated by TAPAS
                           args.output_dir / args.split / 'preprocessed_mention_with_sql_ans.jsonl')
    elif args.data == 'tablefact':
        tf = TableFact(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        tf.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention.jsonl')
    elif args.data == 'wikitq':
        wtq = WikiTQ(args.path)
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
        with open(str(args.path / 'titles_in_3merge.txt'), 'r') as fin:
            for l in fin:
                avoid_titles.add(l.strip())
        turl = TurlData(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        # only use avoid_titles for the test split
        turl.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_cf_avoid3merge.jsonl',
                                      task='cell_filling', avoid_titles=avoid_titles)
        turl.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_sa_avoid3merge.jsonl',
                                      task='schema_augmentation', avoid_titles=avoid_titles)
    elif args.data == 'fakepair':
        find_other_table(args.path, args.output_dir, max_count=3)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
