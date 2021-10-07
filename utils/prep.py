from typing import List
import argparse
import json
from pathlib import Path
import numpy as np


def self_in_dense(ret_file: str):
  mrrs: List[float] = []
  with open(ret_file, 'r') as fin:
    for l in fin:
      idx, bytext, bytable = l.rstrip('\n').split('\t')
      idx = int(idx)
      bytext = [int(s.split(',')[0]) for s in bytext.split(' ') if len(s) > 0]
      bytable = [int(s.split(',')[0]) for s in bytable.split(' ') if len(s) > 0]
      mrr = 0
      for i in range(max(len(bytext), len(bytable))):
        if (i < len(bytext) and bytext[i] == idx) or (i < len(bytable) and bytable[i] == idx):
          mrr = 1 / (i + 1)
          break
      mrrs.append(mrr)
  print(f'avg MRR {np.mean(mrrs)}')


def count_mentions(prep_file: str, max_num_examples: int = 50000):
  num_mentions: List[int] = []
  with open(prep_file, 'r') as fin:
    for i, l in enumerate(fin):
      if i >= max_num_examples:
        break
      num_ment = len(json.loads(l)['context_before_mentions'][0])
      num_mentions.append(num_ment)
  print(f'avg #mention {np.mean(num_mentions)}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, required=True, choices=['self_in_dense', 'count_mentions'])
  parser.add_argument('--inp', type=Path, required=False, nargs='+')
  args = parser.parse_args()

  if args.task == 'self_in_dense':
    self_in_dense(args.inp[0])

  elif args.task == 'count_mentions':
    count_mentions(args.inp[0])
