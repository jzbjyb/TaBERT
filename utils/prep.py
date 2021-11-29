from typing import List, Dict, Set, Tuple, Union
import argparse
import random
import json
import re
import os
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BartForConditionalGeneration
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


def merge_shards(dirs: List[Path],
                 out_dir: Path,
                 epochs: int = 10,
                 keep_shards: List[int] = [-1, -1],
                 skip_first: bool = False,
                 use_softlink: bool = False,
                 alway_use_epochs: List[int] = [None, None]):
  assert len(dirs) == len(keep_shards) == len(alway_use_epochs)
  os.makedirs(str(out_dir), exist_ok=True)
  for e in range(epochs):
    sns: List[int] = [get_shard_num(dir, aue if aue is not None else e) for aue, dir in zip(alway_use_epochs, dirs)]
    keep_sns: List[int] = [sn if keep_shards[i] == -1 else min(sn, keep_shards[i]) for i, sn in enumerate(sns)]
    new_shard_id = 0
    for i, (dir, ksn, aue) in enumerate(zip(dirs, keep_sns, alway_use_epochs)):
      for s in range(ksn):
        from_file = dir / (f'epoch_{aue}.shard{s}.h5' if aue is not None else f'epoch_{e}.shard{s}.h5')
        to_file = out_dir / f'epoch_{e}.shard{new_shard_id}.h5'
        new_shard_id += 1
        if not skip_first or i != 0:
          print(f'{from_file} -> {to_file}')
          if use_softlink:
            from_file_rel = os.path.relpath(str(from_file), str(out_dir))
            os.symlink(str(from_file_rel), str(to_file))
          else:
            copyfile(str(from_file), str(to_file))


def get_table_style() -> str:
  return """
    <head>
      <style>
      table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
      }
      </style>
    </head>
  """


def get_table_html(table: Dict):
  headers = table['header']
  highlight_stype = 'style="color:red;"'
  headers = '<tr>' + ' '.join(
    [f'<th {highlight_stype if h["used"] else ""}>{h["name"]}</th>' for h in headers]) + '</tr>'
  data = table['data']
  data_used = set(map(tuple, table['data_used']))
  data = '<tr>' + '</tr><tr>'.join([' '.join(
    [f'<th {highlight_stype if (row_idx, col_idx) in data_used else ""}>{c}</th>' for col_idx, c in enumerate(row)])
    for row_idx, row in enumerate(data)]) + '</tr>'
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


def get_context_html(context: str, highlight_spans: List[Tuple[int, int]]):
  prev = 0
  context_: List[str] = []
  for s, e in highlight_spans:
    context_.append(context[prev:s])
    context_.append(f'<span style="color:red;">{context[s:e]}</span>')
    prev = e
  context_.append(context[prev:])
  return ''.join(context_)


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
  if len(prep_fins) > 0:
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
    fout.write(get_table_style())
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
        table = get_table_html(id2doc[max_docid]['table'])
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


def visualize_prep_file(prep_file: str, output_file: str, sample_ratio: float = 1.0):
  with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
    fout.write(get_table_style())
    for i, l in tqdm(enumerate(fin)):
      if random.random() > sample_ratio:
        continue
      l = json.loads(l)
      url = get_url(l['uuid']) if 'metadata' not in l else l['metadata']['page_url']
      context = get_context_html(l['context_before'][0], l['context_before_mentions'][0])
      table = l['table']
      table_html = get_table_html(table)
      fout.write(f'<div><a href="{url}">{url}</a><br><h3>{context}</h3></div>\n')
      fout.write(f'{table_html}\n')
      fout.write('<hr>\n')


def replace_context(generation_file: str, prep_file: str, output_file: str, num_files: int, remove_dup: bool = False):
  idx2gens: Dict[int, List[str]] = {}
  num_gens_after_dedup: List[int] = []
  for i in range(num_files):
    with open(f'{generation_file}.{i}', 'r') as fin:
      for l in tqdm(fin):
        l = l.strip().split('\t')
        idx = int(l[-1])
        gens = l[:-3]
        idx2gens[idx] = []
        used: Set[str] = set()
        for gen in gens:
          for rmt in ['<pad>', '<s>', '</s>']:
            gen = gen.replace(rmt, '')
          gen = gen.strip()
          if not remove_dup or gen not in used:
            used.add(gen)
            idx2gens[idx].append(gen)
        num_gens_after_dedup.append(len(used))
  print(f'#sentences after dedup {np.mean(num_gens_after_dedup)}')

  prev_len: List[int] = []
  new_len: List[int] = []
  ids: Set[str] = set()
  with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
    for idx, l in enumerate(tqdm(fin)):
      l = json.loads(l)
      if idx not in idx2gens:
        continue
      prev_len.append(len(l['context_before'][0]))
      assert l['uuid'] not in ids
      ids.add(l['uuid'])
      for gen in idx2gens[idx]:
        l['context_before'] = [gen]
        new_len.append(len(gen))
        fout.write(json.dumps(l) + '\n')

  print(f'#len {np.mean(prev_len)} -> {np.mean(new_len)}')


def process_bidirection(bi_file: str, prep_file: str, output_file: str, num_files: int):
  idx2lp: Dict[int, float] = {}
  for i in range(num_files):
    with open(f'{bi_file}.{i}', 'r') as fin:
      for l in tqdm(fin):
        loss, idx = l.strip().split('\t')
        idx, lp = int(idx), -float(loss)
        if idx >= 15539230:  # TODO: some unknow preprocessing errors
          continue
        idx2lp[idx] = lp

  print(f'#idx in bidirection file {len(idx2lp)} with the max idx {np.max(list(idx2lp.keys()))}')

  prev_uuid = None
  example = None
  context_with_score: List[Tuple[str, Tuple[float, float]]] = []
  num_contexts_to_choose_from: List[int] = []

  def choose(context_with_score, example, fout):
    best_context = sorted(context_with_score, key=lambda x: -sum(x[1]))[0][0]
    example['context_before'] = [best_context]
    fout.write(json.dumps(example) + '\n')

  with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
    for _idx, l in enumerate(tqdm(fin)):
      l = json.loads(l)
      uuid = l['uuid']
      context = l['context_before'][0]
      taidx = _idx * 2
      teidx = _idx * 2 + 1
      if taidx not in idx2lp or teidx not in idx2lp:
        continue
      table2text_lp = idx2lp[taidx]
      text2table_lp = idx2lp[teidx]
      if prev_uuid is not None and uuid != prev_uuid:  # choose the best context
        num_contexts_to_choose_from.append(len(context_with_score))
        choose(context_with_score, example, fout)
        context_with_score = []
      context_with_score.append((context, (table2text_lp, text2table_lp)))
      example = l
      prev_uuid = uuid
    if len(context_with_score) > 0:
      num_contexts_to_choose_from.append(len(context_with_score))
      choose(context_with_score, example, fout)
      context_with_score = []

  print(f'avg #context {np.mean(num_contexts_to_choose_from)}')


def compare_two_files(prep_file1: str, prep_file2: str):
  '''
  prep_file2 might include only a subset from prep_file1
  '''
  lens1: List[int] = []
  lens2: List[int] = []
  with open(prep_file1, 'r') as fin1, open(prep_file2, 'r') as fin2:
    for l2 in tqdm(fin2):
      l1 = fin1.readline()
      if l1 == '': break
      e1 = json.loads(l1)
      e2 = json.loads(l2)
      while e1['uuid'] != e2['uuid']:
        l1 = fin1.readline()
        if l1 == '': break
        e1 = json.loads(l1)
      assert e1['uuid'] == e2['uuid']
      c1 = e1['context_before'][0]
      c2 = e2['context_before'][0]
      #print(c1, '\n', c2)
      lens1.append(len(c1))
      lens2.append(len(c2))
  print(f'compare count {len(lens1)}, avg len {np.mean(lens1)}, {np.mean(lens2)}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, required=True, choices=[
    'self_in_dense', 'count_mentions', 'tapex_ans_in_source', 'merge_shards',
    'ret_compare', 'vis_prep', 'replace_context', 'process_bidirection', 'compare_two_files', 'dump_correct_bart'])
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
    merge_shards(args.inp[:2], args.out, epochs=1, keep_shards=[-1, -1],
                 skip_first=True, use_softlink=False, alway_use_epochs=[None, None])

  elif args.task == 'ret_compare':
    ret_files = args.inp[:-2]
    skip_first = [False, False, False, True, False]
    is_ret_file = [True, True, True, True, False]
    assert len(ret_files) == len(skip_first) == len(is_ret_file)
    ret_query_file, ret_doc_file = args.inp[-2:]
    output_file = args.out
    ret_compare(ret_files, skip_first, is_ret_file, ret_query_file, ret_doc_file, output_file, sample_ratio=0.001, topk=5)

  elif args.task == 'vis_prep':
    prep_file = args.inp[0]
    output_file = args.out
    visualize_prep_file(prep_file, output_file, sample_ratio=0.01)

  elif args.task == 'replace_context':
    generation_file, prep_file = args.inp
    output_file = args.out
    num_gpu = 8
    replace_context(generation_file, prep_file, output_file, num_files=num_gpu, remove_dup=False)

  elif args.task == 'process_bidirection':
    bi_file, prep_file = args.inp
    output_file = args.out
    num_gpu = 64
    process_bidirection(bi_file, prep_file, output_file, num_files=num_gpu)

  elif args.task == 'compare_two_files':
    prep_file1, prep_file2 = args.inp
    compare_two_files(prep_file1, prep_file2)

  elif args.task == 'dump_correct_bart':
    # the mask embedding in old verions of BART-base is incorrect
    # use the latest version of transformers when running this function
    model_name = 'facebook/bart-large'
    dump_dir = '/mnt/root/TaBERT/data/runs/bart_large'
    model = BartForConditionalGeneration.from_pretrained(model_name).eval()
    save_function = lambda obj, f: torch.save(obj, f, _use_new_zipfile_serialization=False)
    model.save_pretrained(dump_dir, save_function=save_function)
