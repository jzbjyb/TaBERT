from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
import torch
import numpy as np
from table_bert import TableBertModel
from table_bert.utils import compute_mrr
from table_bert.spider import Spider
from table_bert.ranker import Ranker


def main():
    parser = ArgumentParser()
    parser.add_argument('--sample_file', type=Path, required=True)
    parser.add_argument('--db_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--group_by', type=int, default=10)
    args = parser.parse_args()

    model = Ranker(TableBertModel.from_pretrained(str(args.model_path)))
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    model = model.to(device)
    model.eval()

    db2table2column = json.load(open(args.db_file, 'r'))
    questions = []
    tables = []
    scores = []
    labels = []
    with open(args.sample_file, 'r') as fin:
        with torch.no_grad():
            for l in tqdm(fin):
                question, db_id, table_name, label = l.strip().split('\t')
                table = Spider.convert_to_table(table_name, db2table2column[db_id][table_name])
                questions.append(model.tokenizer.tokenize(question))
                tables.append(table.tokenize(model.tokenizer))
                labels.append(int(label))
                if len(questions) >= args.batch_size:
                    score = model(questions, tables)
                    scores.extend(score.cpu().numpy().tolist())
                    questions = []
                    tables = []
            if len(questions) > 0:
                score = model(questions, tables)
                scores.extend(score.cpu().numpy().tolist())

    mrrs = []
    with open(args.output_file, 'w') as fout:
        for i in range(0, len(labels), args.group_by):
            ls = labels[i:i + args.group_by]
            ss = scores[i:i + args.group_by]
            mrrs.append(compute_mrr(ss, ls))
            for s in ss:
                fout.write('{}\n'.format(s))

    print('MRR {}'.format(np.mean(mrrs)))


if __name__ == '__main__':
    main()
