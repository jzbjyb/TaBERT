from argparse import ArgumentParser
from pathlib import Path
from table_bert.spider import Spider


def main():
    parser = ArgumentParser()
    parser.add_argument('--spider_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()

    spider = Spider(args.spider_path)
    spider.convert_to_tabert_format(args.output_dir / 'db_tabert.json')
    spider.sample_negative(args.output_dir / 'samples.tsv', neg_num=9)

if __name__ == '__main__':
    main()
