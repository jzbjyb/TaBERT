from argparse import ArgumentParser
from pathlib import Path
import os
import random
import numpy as np
from table_bert.totto import Totto
from table_bert.wikisql import WikiSQL
from table_bert.tablefact import TableFact
from table_bert.wikitablequestions import WikiTQ
from table_bert.turl import TurlData


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['totto', 'wikisql', 'tablefact', 'wtq', 'turl', 'overlap'])
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
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
