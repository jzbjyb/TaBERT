from argparse import ArgumentParser
from pathlib import Path
import os
import random
from table_bert.totto import Totto
from table_bert.wikisql import WikiSQL
from table_bert.tablefact import TableFact


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['totto', 'wikisql', 'tablefact'])
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    random.seed(2021)

    if args.data == 'totto':
        totto = Totto(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        totto.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')
    elif args.data == 'wikisql':
        wsql = WikiSQL(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        wsql.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')
    elif args.data == 'tablefact':
        tf = TableFact(args.path)
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        tf.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
