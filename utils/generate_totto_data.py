from argparse import ArgumentParser
from pathlib import Path
import os
import random
from table_bert.totto import Totto


def main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    random.seed(2021)

    totto = Totto(args.path)
    os.makedirs(args.output_dir / args.split, exist_ok=True)
    totto.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')


if __name__ == '__main__':
    main()
