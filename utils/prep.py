from typing import List, Dict, Set, Tuple, Union
import argparse
import random
import json
import re
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from table_bert.dataset_utils import BasicDataset
from table_bert.utils import get_url


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


def count_mentions(prep_file: str, max_num_examples: Union[int, None] = 50000):
  num_mentions: List[int] = []
  context_lens: List[int] = []
  with open(prep_file, 'r') as fin:
    for i, l in tqdm(enumerate(fin)):
      if max_num_examples and i >= max_num_examples:
        break
      l = json.loads(l)
      nm = len(l['context_before_mentions'][0])
      cl = len(l['context_before'][0])
      num_mentions.append(nm)
      context_lens.append(cl)
  print(f'avg #mention {np.mean(num_mentions)}, avg context len {np.mean(context_lens)}')
  plt.hist(num_mentions, bins=100, weights=np.ones(len(num_mentions)) / len(num_mentions))
  plt.savefig('test_num_mentions.png')
  plt.hist(context_lens, bins=100, weights=np.ones(len(context_lens)) / len(context_lens))
  plt.savefig('test_context_lens.png')


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


def visualize_table(table: Dict):
  headers = [h['name'] for h in table['header']]
  headers = '<tr>' + ' '.join([f'<th>{h}</th>' for h in headers]) + '</tr>'
  data = table['data']
  data = '<tr>' + '</tr><tr>'.join([' '.join([f'<th>{c}</th>' for c in row]) for row in data]) + '</tr>'
  table_str = f"""
    <table>
      <thead>
        {headers}
      </thead>
      <tbody>
        {data}
      </tbody>
    </table>
  """
  return table_str


def ret_compare(ret_files: List[str],
                skip_first: List[bool],
                is_ret: List[bool],
                ret_query_file: str,
                ret_doc_file: str,
                output_file: str,
                sample_ratio: float = 1.0,
                topk: int = 1):
  is_same = ret_query_file == ret_doc_file

  print('read retrieval file ...')
  qid2docids: Dict[int, List[List[int]]] = defaultdict(list)
  all_qids: Set[int] = set()
  all_docids: Set[int] = set()
  ret_fins = [open(rf, 'r') for ir, rf in zip(is_ret, ret_files) if ir]
  try:
    while True:
      try:
        lines: List[str] = [rf.readline() for rf in ret_fins]
        if lines[0] == '':
          break
        if random.random() > sample_ratio:
          continue
        for sf, line in zip(skip_first, lines):
          qid, docs = line.rstrip('\n').split('\t')[:2]
          qid = int(qid)
          docids = [int(d.split(',')[0]) for d in docs.split(' ')]
          if is_same and sf:  # skip the first retrieved doc which is always self
            docids = docids[1:]
          docids = docids[:topk]
          qid2docids[qid].append(docids)
          all_qids.add(qid)
          all_docids.update(docids)
      except StopIteration:
        break
  finally:
    for rf in ret_fins:
      if rf:  rf.close()

  print('read prep file ...')
  context2table: List[List[Tuple[str, Dict]]] = []
  prep_fins = [open(rf, 'r') for ir, rf in zip(is_ret, ret_files) if not ir]
  try:
    while True:
      try:
        lines: List[str] = [pf.readline() for pf in prep_fins]
        if lines[0] == '':
          break
        if random.random() > sample_ratio:
          continue
        context2table.append([])
        for line in lines:
          line = json.loads(line)
          context2table[-1].append((line['context_before'][0], line['table']))
      except StopIteration:
        break
  finally:
    for pf in prep_fins:
      if pf:  pf.close()

  print('read query/doc files ...')
  id2query: Dict[str, Dict] = {}
  with open(ret_query_file, 'r') as fin:
    for i, l in tqdm(enumerate(fin)):
      if i not in all_qids and i not in all_docids:
        continue
      id2query[i] = json.loads(l)
  if not is_same:
    id2doc: Dict[str, Dict] = {}
    with open(ret_doc_file, 'r') as fin:
      for i, l in enumerate(fin):
        if i not in all_docids:
          continue
        id2doc[i] = json.loads(l)
  else:
    id2doc = id2query

  print('output ...')
  with open(output_file, 'w') as fout:
    fout.write("""
      <head>
        <style>
        table, th, td {
          border: 1px solid black;
          border-collapse: collapse;
        }
        </style>
      </head>
    """)
    fout.write('<body>\n')
    qid2docids: List[Tuple] = list(qid2docids.items())

    group2nm: Dict[int, List[int]] = defaultdict(list)
    # analyze retrieval files
    for i in tqdm(np.random.permutation(len(qid2docids))):
      qid, docids = qid2docids[i]
      context_url = get_url(id2query[qid]['uuid'])
      context = id2query[qid]['context_before'][0]
      fout.write(f'<div><a href="{context_url}">{context_url}</a><br><h3>{context}</h3></div>\n')
      for group, _docids in enumerate(docids):
        # find the one with highest overlap
        max_nm = 0
        max_docid = None
        for docid in _docids:
          num_mentions = len(BasicDataset.get_mention_locations(context, id2doc[docid]['table']['data'])[0])
          if num_mentions >= max_nm:
            max_nm = num_mentions
            max_docid = docid
        table_url = get_url(id2doc[max_docid]['uuid'])
        table = visualize_table(id2doc[max_docid]['table'])
        group2nm[group].append(max_nm)
        fout.write(f'<div><a href="{table_url}">{table_url}</a><br>{table}</div><br>\n')
      fout.write('<hr>\n')
    fout.write('</body>')

    # analyze prep files
    for _context2table in context2table:
      for group, (context, table) in enumerate(_context2table):
        nm = len(BasicDataset.get_mention_locations(context, table['data'])[0])
        group2nm[group + len(ret_fins)].append(nm)

    print(f'total count {len(qid2docids)}')
    group2nm = {k: (np.mean(v), len(v)) for k, v in group2nm.items()}
    print(f'group2nm: {group2nm}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, required=True, choices=[
    'self_in_dense', 'count_mentions', 'tapex_ans_in_source', 'merge_shards', 'ret_compare'])
  parser.add_argument('--inp', type=Path, required=False, nargs='+')
  parser.add_argument('--out', type=Path, required=False)
  args = parser.parse_args()

  SEED = 2021
  random.seed(SEED)
  np.random.seed(SEED)

  if args.task == 'self_in_dense':
    self_in_dense(args.inp[0])

  elif args.task == 'count_mentions':
    count_mentions(args.inp[0], max_num_examples=None)

  elif args.task == 'tapex_ans_in_source':
    tapex_ans_in_source(args.inp[0])

  elif args.task == 'merge_shards':
    merge_shards(args.inp[:2], args.out, epochs=10, keep_shards=[-1, 2], skip_first=True)

  elif args.task == 'ret_compare':
    ret_files = args.inp[:-2]
    skip_first = [False, False, False, True, False]
    is_ret_file = [True, True, True, True, False]
    assert len(ret_files) == len(skip_first) == len(is_ret_file)
    ret_query_file, ret_doc_file = args.inp[-2:]
    output_file = args.out
    ret_compare(ret_files, skip_first, is_ret_file, ret_query_file, ret_doc_file, output_file, sample_ratio=0.001, topk=5)
