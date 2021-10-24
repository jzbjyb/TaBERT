from typing import List
import argparse
import json
import re
from shutil import copyfile
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


def tapex_ans_in_source(pred_file: str):
  ins = []
  with open(pred_file, 'r') as fin:
    for l in fin:
      l = json.loads(l)
      source = [h['name'] for h in l['table']['header']]
      source += [c for r in l['table']['data'] for c in r]
      source = ' '.join(source).lower()
      anss = l['answers']
      _in = True
      for ans in anss:
        if ans.lower() not in source:
          _in = False
          break
      ins.append(_in)
  print(f'answer in source: {np.mean(ins)}')


def get_shard_num(dir: Path, epoch: int) -> int:
  epoch_prefix = dir / f'epoch_{epoch}'
  shard_files = list(epoch_prefix.parent.glob(epoch_prefix.name + '.shard*.h5'))
  shard_ids = [int(re.search(r'shard(\d+)', str(f)).group(1)) for f in shard_files]
  shard_num = max(shard_ids) + 1
  return shard_num


def merge_shards(dirs: List[Path], out_dir: Path, epochs: int = 10, keep_shards: List[int] = [-1, -1], skip_first: bool = False):
  assert len(dirs) == len(keep_shards)
  for e in range(epochs):
    sns: List[int] = [get_shard_num(dir, e) for dir in dirs]
    keep_sns: List[int] = [sn if keep_shards[i] == -1 else min(sn, keep_shards[i]) for i, sn in enumerate(sns)]
    new_shard_id = 0
    for i, (dir, ksn) in enumerate(zip(dirs, keep_sns)):
      for s in range(ksn):
        from_file = dir / f'epoch_{e}.shard{s}.h5'
        to_file = out_dir / f'epoch_{e}.shard{new_shard_id}.h5'
        new_shard_id += 1
        if not skip_first or i != 0:
          print(f'{from_file} -> {to_file}')
          copyfile(str(from_file), str(to_file))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, required=True, choices=[
    'self_in_dense', 'count_mentions', 'tapex_ans_in_source', 'merge_shards'])
  parser.add_argument('--inp', type=Path, required=False, nargs='+')
  parser.add_argument('--out', type=Path, required=False)
  args = parser.parse_args()

  if args.task == 'self_in_dense':
    self_in_dense(args.inp[0])

  elif args.task == 'count_mentions':
    count_mentions(args.inp[0])

  elif args.task == 'tapex_ans_in_source':
    tapex_ans_in_source(args.inp[0])

  elif args.task == 'merge_shards':
    merge_shards(args.inp[:2], args.out, epochs=10, keep_shards=[-1, 2], skip_first=True)
