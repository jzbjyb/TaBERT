from typing import List, Dict, Tuple, Set
import argparse
from pathlib import Path
import json
import random
from tqdm import tqdm
from collections import defaultdict
import functools
import time
import sling
import re
import numpy as np
from multiprocessing import Process, Queue
from table_bert.utils import get_url
from table_bert.dataset_utils import BasicDataset
from table_bert.wikidata import topic2categories, WikipediaCategory


def get_commons_docschema():
  commons = sling.Store()
  docschema = sling.DocumentSchema(commons)
  commons.freeze()
  return commons, docschema
self_commons, self_docschema = get_commons_docschema()


only_alphanumeric = re.compile('[\W_]+')


def load_url2tables(prep_file: Path) -> Tuple[Dict[str, List], List[Dict]]:
  table_count = image_url_count = 0
  url2tables: Dict[str, List] = defaultdict(list)
  raw_examples: List[Dict] = []
  with open(str(prep_file), 'r') as fin:
    for l in tqdm(fin):
      l = json.loads(l)
      table = l['table']
      url = get_url(l['uuid'])
      if url.endswith(('.jpg', '.jpeg', '.svg', '.gif', '.png', '.eps', '.bmp', '.tif', '.tiff')):
        image_url_count += 1
        continue
      url2tables[url].append(table)
      raw_examples.append(l)
      table_count += 1
  url_count = len(url2tables)
  print(f'#url {url_count}, #table {table_count}, #image url {image_url_count}')
  return url2tables, raw_examples


def load_pageid2tables(prep_file: Path) -> Dict[str, List]:
  table_count = 0
  pageid2tables: Dict[str, List] = defaultdict(list)
  with open(str(prep_file), 'r') as fin:
    for l in tqdm(fin):
      l = json.loads(l)
      table = l['table']
      pageid = l['pageid']
      pageid2tables[pageid].append(table)
      table_count += 1
  pageid_count = len(pageid2tables)
  print(f'#pageid {pageid_count}, #table {table_count}')
  return pageid2tables


def match_raw_worker(input_queue, output_queue):
  while True:
    args = input_queue.get()
    if type(args) is str and args == 'DONE':
      break
    examples = args['examples']
    for example in examples:
      context = example['context_before'][0]
      table = example['table']
      locations, location2cells = BasicDataset.get_mention_locations(context, table['data'])
      mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
      data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))

      table['data_used'] = data_used
      example = {
        'uuid': example['uuid'],
        'table': table,
        'context_before': [context],
        'context_before_mentions': [locations],
        'context_before_mentions_cells': [mention_cells],
        'context_after': []
      }
      output_queue.put(example)


def match_raw(examples: List[Dict],
              output_file: str,
              batch_size: int = 16,
              num_threads: int = 1,
              log_interval: int = 100):
  input_queue = Queue()
  output_queue = Queue()
  processes = []

  # start processes
  for _ in range(num_threads):
    p = Process(target=match_raw_worker,
                args=(input_queue, output_queue))
    p.daemon = True
    p.start()
    processes.append(p)
  write_p = Process(target=functools.partial(SlingExtractor.write_worker, log_interval=log_interval),
                    args=(output_file, output_queue))
  write_p.start()

  for i in range(0, len(examples), batch_size):
    batch = examples[i:i + batch_size]
    input_queue.put({'examples': batch})

  for _ in processes:
    input_queue.put('DONE')
  for p in processes:
    p.join()
  output_queue.put('DONE')
  write_p.join()


class SlingExtractor(object):
  def __init__(self, root_dir: Path, num_splits: int = 10):
    self.root_dir = root_dir
    self.num_splits = num_splits
    self.lang = 'en'

  @staticmethod
  def get_metadata(frame: sling.Frame) -> Dict:
    pageid = str(frame.get('/wp/page/pageid'))  # wikiPEDIA page ID
    title = frame.get('/wp/page/title')  # article title
    item = frame.get('/wp/page/item')  # wikidata ID associated to the article
    url = frame.get('url')
    return {
      'pageid': pageid,
      'title': title,
      'item': item,
      'url': url,
    }

  @staticmethod
  def get_categories(frame: sling.Frame) -> List[str]:
    return [v.id for k, v in frame if k.id == '/wp/page/category']

  @staticmethod
  def get_linked_entity(mention):
    if 'evokes' not in mention.frame:
      return None
    if type(mention.frame['evokes']) != sling.Frame:
      return None
    if 'is' in mention.frame['evokes']:
      if type(mention.frame['evokes']['is']) != sling.Frame:
        if ('isa' in mention.frame['evokes'] and
          mention.frame['evokes']['isa'].id == '/w/time' and
          type(mention.frame['evokes']['is']) == int):
          return mention.frame['evokes']['is']
        else:
          return None
      else:
        return mention.frame['evokes']['is'].id
    return mention.frame['evokes'].id

  @staticmethod
  def split_document_by_sentence(tokens) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]:
    # build token maps
    tok_to_sent_id, tok_to_para_id, sent_to_span, para_to_span = {}, {}, {}, {}
    i = sent_begin = para_begin = 0
    sent_id = para_id = 0
    for i, token in enumerate(tokens):
      if i > 0 and token.brk == 4:  # new para
        para_to_span[para_id] = (para_begin, i)
        sent_to_span[sent_id] = (sent_begin, i)
        para_id += 1
        sent_id += 1
        sent_begin = para_begin = i
      elif i > 0 and token.brk == 3:  # new sentence
        sent_to_span[sent_id] = (sent_begin, i)
        sent_id += 1
        sent_begin = i
      tok_to_sent_id[i] = sent_id
    para_to_span[para_id] = (para_begin, i + 1)
    sent_to_span[sent_id] = (sent_begin, i + 1)
    return tok_to_sent_id, sent_to_span

  @staticmethod
  def annotate_sentence(urls: List[str],
                        documents: List[bytes],
                        tables_li: List[List[Dict]],
                        use_all_more_than_num_mentions: int,
                        topk: int) -> List[Dict]:
    examples: List[Dict] = []

    for url, document, tables in zip(urls, documents, tables_li):
      # parse document
      store = sling.Store(self_commons)
      doc_frame = store.parse(document)
      document = sling.Document(doc_frame, store, self_docschema)
      tokens = [t.word for t in document.tokens]
      token_brks = [t.brk for t in document.tokens]
      tokens = [t if tb == 0 else (' ' + t) for t, tb in zip(tokens, token_brks)]  # add space if needed

      # split by sentence
      tok_to_sent_id, sent_to_span = SlingExtractor.split_document_by_sentence(document.tokens)

      for index, table in enumerate(tables):
        kept_sent_with_mentions: List[Tuple[str, List[Tuple], List[List[Tuple]]]] = []
        # find the topk best-matching sentence
        local_num_context_count = 0
        for sent_id, (sent_start, sent_end) in sent_to_span.items():
          if sent_end - sent_start > 512:
            # rule 0: skip long sentence
            continue
          sent = ''.join(tokens[sent_start:sent_end])
          if not sent.endswith(('!', '"', "'", '.', ':', ';', '?')):
            # rule 1: skip text without punctuations at the end (e.g., table snippets)
            continue
          sent_only_alphanumeric = only_alphanumeric.sub('', sent)
          sent_only_numberic = re.sub('[^0-9]', '', sent_only_alphanumeric)
          number_ratio = len(sent_only_numberic) / (len(sent_only_alphanumeric) or 1)
          if number_ratio > 0.8:
            # rule 2: skip text where most characters are numeric characters
            continue
          locations, location2cells = BasicDataset.get_mention_locations(sent, table['data'])
          mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
          data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))
          cover_ratio = np.sum([e - s for s, e in locations]) / (len(sent) or 1)
          if cover_ratio > 0.8:
            # rule 3: skip text that has over 80% of overlap with the table, which is likely to be table snippets
            continue
          if not use_all_more_than_num_mentions:
            kept_sent_with_mentions.append((sent, locations, mention_cells))
          elif use_all_more_than_num_mentions and len(locations) >= use_all_more_than_num_mentions:
            table['data_used'] = data_used
            examples.append({
              'uuid': f'sling_{url}_{index}_{local_num_context_count}',
              'table': table,
              'context_before': [sent],
              'context_before_mentions': [locations],
              'context_before_mentions_cells': [mention_cells],
              'context_after': []
            })
            local_num_context_count += 1

        if len(kept_sent_with_mentions) <= 0:
          continue

        if not use_all_more_than_num_mentions:  # output topk
          # sort by number of mentions
          kept_sent_with_mentions = sorted(kept_sent_with_mentions, key=lambda x: -len(x[1]))[:topk]
          merge_sent: str = ''
          merge_locations: List[Tuple] = []
          merge_mention_cells: List[List[Tuple]] = []
          for sent, locations, mention_cells in kept_sent_with_mentions:
            offset = len(merge_sent) + int(len(merge_sent) > 0)
            merge_sent = (merge_sent + ' ' + sent) if len(merge_sent) > 0 else sent
            merge_locations.extend([(s + offset, e + offset) for s, e in locations])
            merge_mention_cells.extend(mention_cells)
          data_used = sorted(list(set(mc for mcs in merge_mention_cells for mc in mcs)))
          table['data_used'] = data_used
          examples.append({
            'uuid': f'sling_{url}_{index}',
            'table': table,
            'context_before': [merge_sent],
            'context_before_mentions': [merge_locations],
            'context_before_mentions_cells': [merge_mention_cells],
            'context_after': []
          })
    return examples

  @staticmethod
  def match_sentence_with_table_worker(input_queue: Queue,
                                       output_queue: Queue,
                                       use_all_more_than_num_mentions: int,
                                       topk: int):
    func = functools.partial(SlingExtractor.annotate_sentence,
                             use_all_more_than_num_mentions=use_all_more_than_num_mentions, topk=topk)
    while True:
      args = input_queue.get()
      if type(args) is str and args == 'DONE':
        break
      for example in func(**args):
        output_queue.put(example)

  @staticmethod
  def write_worker(output_file: str, output_queue: Queue, log_interval):
    sent_len_li: List[int] = []
    num_mentions_li: List[int] = []
    empty_count = non_empty_count = i = 0
    with open(output_file, 'w') as fout, tqdm() as pbar:
      while True:
        example = output_queue.get()
        i += 1
        if type(example) is str and example == 'DONE':
          break
        pbar.update(1)
        nm = len(example['context_before_mentions'][0])
        if nm <= 0:
          empty_count += 1
          continue
        non_empty_count += 1
        sent_len_li.append(len(example['context_before'][0]))
        num_mentions_li.append(nm)
        fout.write(json.dumps(example) + '\n')
        if i > 0 and i % log_interval == 0:
          print(f'avg sent len {np.mean(sent_len_li)}, '
                f'avg #mentions {np.mean(num_mentions_li)} '
                f'empty/non-empty {empty_count}/{non_empty_count}')

  def match_sentence_with_table(self,
                                url2tables: Dict[str, List],
                                key_feild: str,
                                from_split: int,  # inclusive
                                to_split: int,  # exclusive
                                output_file: str,
                                use_all_more_than_num_mentions: int = 0,
                                topk: int = 1,
                                batch_size: int = 16,
                                num_threads: int = 1,
                                log_interval: int = 5000):

    input_queue = Queue()
    output_queue = Queue()
    processes = []

    # start processes
    for _ in range(num_threads):
      p = Process(target=functools.partial(self.match_sentence_with_table_worker,
                                           use_all_more_than_num_mentions=use_all_more_than_num_mentions,
                                           topk=topk),
                  args=(input_queue, output_queue))
      p.daemon = True
      p.start()
      processes.append(p)
    write_p = Process(target=functools.partial(self.write_worker, log_interval=log_interval),
                      args=(output_file, output_queue))
    write_p.start()

    # read data
    coverted_urls: Set[str] = set()
    urls: List[str] = []
    doc_raws: List[bytes] = []
    tables_li: List[List[Dict]] = []
    for sp in range(from_split, to_split):
      corpus = sling.Corpus(str(self.root_dir / self.lang / f'documents-0000{sp}-of-00010.rec'))
      for n, (doc_id, doc_raw) in tqdm(enumerate(corpus.input), disable=True):
        doc_id = doc_id.decode('utf-8')
        store = sling.Store(self_commons)
        doc_frame = store.parse(doc_raw)
        metadata = self.get_metadata(doc_frame)
        url = metadata[key_feild]

        if url not in url2tables:
          continue
        if url in coverted_urls:
          print(f'{url} shows multiple times')
        coverted_urls.add(url)

        urls.append(url)
        doc_raws.append(doc_raw)
        tables_li.append(url2tables[url])

        if len(urls) >= batch_size:
          input_queue.put({'urls': urls, 'documents': doc_raws, 'tables_li': tables_li})
          urls = []
          doc_raws = []
          tables_li = []

    for _ in processes:
      input_queue.put('DONE')
    for p in processes:
      p.join()
    output_queue.put('DONE')
    write_p.join()

    print(f'#coverted urls {len(coverted_urls)}, #total urls {len(url2tables)}')
    print(f'uncovered urls {list(set(url2tables.keys()) - coverted_urls)[:10]}')

  def build_category_hierarchy(self, out_file: str):
    child2parents: Dict[str, List[str]] = {}
    for sp in range(0, 10):
      corpus = sling.Corpus(str(self.root_dir / self.lang / f'category-documents-0000{sp}-of-00010.rec'))
      for n, (wid, doc_raw) in tqdm(enumerate(corpus.input)):
        wid = wid.decode('utf-8')
        store = sling.Store(self_commons)
        doc_frame = store.parse(doc_raw)
        categories = self.get_categories(doc_frame)
        assert wid not in child2parents
        child2parents[wid] = categories
    with open(out_file, 'w') as fout:
      for c, ps in child2parents.items():
        fout.write('{}\t{}\n'.format(c, ','.join(ps)))

  @classmethod
  def get_top_category_by_request(cls,
                                  pageids: List[str],
                                  wc: WikipediaCategory,
                                  top_categories: Set[str],
                                  out_file: str = None):
    pageid2cates = wc.pageid2cate(pageids, max_limit=1)

    all_cates = list(set(c for cates in pageid2cates.values() for c in cates))
    print(f'#categories {len(all_cates)}')
    title2wid = wc.titles2other(all_cates, field='wikidata')

    out_file = out_file or '/tmp/test'
    with open(out_file, 'w') as fout:
      for pageid, cates in pageid2cates.items():
        cates = [title2wid[c] for c in cates if c in title2wid]
        try:
          if len(cates) <= 0:
            raise Exception('not categories as wikidata ids')
          top_category = wc.find_closest_top_category(cates, top_categories)
          top_category: str = ','.join(top_category)
          fout.write(f'{pageid}\t{top_category}\n')
        except KeyboardInterrupt as e:
          raise e
        except:
          print(f'error {pageid}')


  def get_top_category(self,
                       pageids: Set[str],
                       wc: WikipediaCategory,
                       top_categories: Set[str],
                       from_split: int = 0,
                       to_split: int = 10,
                       out_file: str = None):
    out_file = out_file or '/tmp/test'
    pageid2topcate: Dict[str, str] = {}
    with open(out_file, 'w') as fout:
      for sp in range(from_split, to_split):
        corpus = sling.Corpus(str(self.root_dir / self.lang / f'documents-0000{sp}-of-00010.rec'))
        for n, (doc_id, doc_raw) in tqdm(enumerate(corpus.input), disable=False):
          doc_id = doc_id.decode('utf-8')
          store = sling.Store(self_commons)
          doc_frame = store.parse(doc_raw)
          metadata = self.get_metadata(doc_frame)
          pageid = metadata['pageid']
          if pageid not in pageids:
            continue
          categories = self.get_categories(doc_frame)
          try:
            top_category = wc.find_closest_top_category(categories, top_categories)
            pageid2topcate[pageid] = top_category
            top_category: str = ','.join(top_category)
            fout.write(f'{pageid}\t{top_category}\n')
          except KeyboardInterrupt as e:
            raise e
          except:
            print(f'error {pageid}')
    return pageid2topcate


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=[
    'url_as_key', 'pageid_as_key', 'match_raw', 'build_category_hierarchy', 'pageid2topic', 'assign_topic'], default='url_as_key')
  parser.add_argument('--inp', type=Path, nargs='+')
  parser.add_argument('--out', type=Path)
  parser.add_argument('--topk', type=int, default=1, help='number of sentences used as context')
  parser.add_argument('--threads', type=int, default=40)
  args = parser.parse_args()

  SEED = 2021
  random.seed(SEED)
  np.random.seed(SEED)

  if args.task == 'url_as_key':
    prep_file, sling_root = args.inp
    out_file = args.out

    url2tables = load_url2tables(prep_file)[0]
    se = SlingExtractor(sling_root)
    se.match_sentence_with_table(
      url2tables, key_feild='url', from_split=0, to_split=10, output_file=out_file,
      use_all_more_than_num_mentions=0, topk=args.topk, batch_size=16, num_threads=args.threads, log_interval=5000)

  elif args.task == 'pageid_as_key':
    prep_file, sling_root = args.inp
    out_file = args.out

    pageid2tables = load_pageid2tables(prep_file)
    se = SlingExtractor(sling_root)
    se.match_sentence_with_table(
      pageid2tables, key_feild='pageid', from_split=0, to_split=10, output_file=out_file,
      use_all_more_than_num_mentions=1, topk=args.topk, batch_size=16, num_threads=args.threads, log_interval=100)

  elif args.task == 'match_raw':
    prep_file = args.inp[0]
    out_file = args.out

    url2tables, raw_examples = load_url2tables(prep_file)
    match_raw(raw_examples, output_file=out_file, batch_size=16, num_threads=args.threads, log_interval=5000)

  elif args.task == 'build_category_hierarchy':
    sling_root = args.inp[0]
    out_file = args.out
    se = SlingExtractor(sling_root)
    se.build_category_hierarchy(out_file)

  elif args.task == 'pageid2topic':
    find_missing_from_sling = 2
    wikipedia_cate_hierachy = 'data/wikipedia_category/sling_category_child2parents.txt'

    if find_missing_from_sling:
      prep_files = args.inp[:-find_missing_from_sling]
      pageid2cate_files = args.inp[-find_missing_from_sling:]
    else:
      prep_files = args.inp[:-1]
      sling_root = args.inp[-1]
    out_file = args.out

    pids: Set[str] = set()
    for prep_file in prep_files:
      with open(prep_file, 'r') as fin:
        for l in fin:
          pid = json.loads(l)['pageid']
          pids.add(pid)

    if find_missing_from_sling:
      found_pids: Set[str] = set()
      for pageid2cate_file in pageid2cate_files:
        with open(pageid2cate_file, 'r') as fin:
          for l in fin:
            pageid, cates = l.rstrip('\n').split('\t')
            found_pids.add(pageid)

    wc = WikipediaCategory(child2parent_file=wikipedia_cate_hierachy, format='sling')
    top_categories: Set[str] = set(c for cs in topic2categories.values() for c in cs)
    print(f'#page ids {len(pids)}')

    if find_missing_from_sling:
      miss_pids = list(pids - found_pids)
      print(f'#missing page ids {len(miss_pids)}')
      SlingExtractor.get_top_category_by_request(miss_pids, wc, top_categories, out_file=out_file)
    else:
      se = SlingExtractor(sling_root)
      pageid2category: Dict[str, str] = se.get_top_category(
        pids, wc, top_categories, from_split=0, to_split=10, out_file=out_file)

  elif args.task == 'assign_topic':
    pageid2cate_files = args.inp
    out_file = args.out

    cate2topic: Dict[str, str] = {}
    for topic, cates in topic2categories.items():
      for cate in cates:
        cate2topic[cate] = topic

    pageid2cates: Dict[str, List[str]] = {}
    pageid2topic: Dict[str, str] = {}
    topic2count: Dict[str, int] = defaultdict(lambda: 0)
    with open(out_file, 'w') as fout:
      for pageid2cate_file in pageid2cate_files:
        with open(pageid2cate_file, 'r') as fin:
          for l in fin:
            pageid, cates = l.rstrip('\n').split('\t')
            cates = cates.split(',')
            pageid2cates[pageid] = cates
            topics, counts = np.unique([cate2topic[cate] for cate in cates], return_counts=True)
            topic = topics[0]
            pageid2topic[pageid] = topic
            topic2count[topic] += 1
            fout.write(f'{pageid}\t{topic}\n')

    print('topic2count', sorted(topic2count.items(), key=lambda x: -x[1]))
