from argparse import ArgumentParser
from pathlib import Path
import os
import json
from table_bert.spider import Spider
from table_bert.squall import Squall


def main():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['spider', 'squall', 'tapex', 'combine_nl'])
    parser.add_argument('--path', type=Path, required=True, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    if args.task == 'spider':
        spider_dir = args.path[0]
        spider = Spider(spider_dir)
        os.makedirs(args.output / args.split, exist_ok=True)
        spider.gen_sql2nl_data(args.split, args.output / args.split / 'sqlnl.json')
        #spider.convert_to_tabert_format(args.split, args.output / args.split / 'db_tabert.json')
        #spider.sample_negative(args.split, args.output / args.split / 'samples.tsv', neg_num=9)

    elif args.task == 'squall':
        squall_file, wtq_dir = args.path
        output_path = args.output
        fews = [16, 32, 64, 128, 256, 512, 1024]
        squall = Squall(squall_file)
        for few in fews:
            print(f'load the sql for few-shot {few}')
            squall.get_subset(wtq_dir / f'train.src.{few}', output_path / f'train.src.{few}')
        squall.get_subset(wtq_dir / f'train.src', output_path / f'train.src')
        squall.get_subset(wtq_dir / f'valid.src', output_path / f'valid.src')
        squall.get_subset(wtq_dir / f'test.src', output_path / f'test.src')

    elif args.task == 'tapex':
        prep_file = args.path[0]
        out = args.output
        with open(prep_file, 'r') as fin, open(out, 'w') as fout:
            for l in fin:
                l = json.loads(l)
                sql = l['context_before'][0]
                nl = 'dummy'
                td = {
                    'uuid': l['uuid'],
                    'metadata': {
                        'sql': sql,
                        'nl': nl,
                    },
                    'table': {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []},
                    'context_before': [nl],
                    'context_after': []
                }
                fout.write(json.dumps(td) + '\n')


if __name__ == '__main__':
    main()
