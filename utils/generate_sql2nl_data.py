from argparse import ArgumentParser
from pathlib import Path
import os
import json
from table_bert.spider import Spider


def main():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['spider', 'tapex', 'combine_nl'])
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()

    if args.task == 'spider':
        spider = Spider(args.path)
        os.makedirs(args.output / args.split, exist_ok=True)
        spider.gen_sql2nl_data(args.split, args.output / args.split / 'sqlnl.json')
        #spider.convert_to_tabert_format(args.split, args.output / args.split / 'db_tabert.json')
        #spider.sample_negative(args.split, args.output / args.split / 'samples.tsv', neg_num=9)
    elif args.task == 'tapex':
        prep_file = args.path
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
    elif args.task == 'combine_nl':
        nl_file, prep_file = args.path
        out = args.output

        with open(nl_file, 'r') as fin:



if __name__ == '__main__':
    main()
