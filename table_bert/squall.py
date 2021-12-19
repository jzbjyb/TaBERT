from typing import Dict, Set
import json
from tqdm import tqdm
from pathlib import Path
from table_bert.dataset_utils import BasicDataset


class Squall(BasicDataset):
    def __init__(self, json_file: Path):
        self.ntid2example = self.load(json_file)

    @staticmethod
    def load(filepath: Path) -> Dict[str, Dict]:
      ntid2example: Dict[str, Dict] = {}
      with open(filepath, 'r') as fin:
        data = json.load(fin)
        for example in data:
          ntid = example['nt']
          nl: str = ' '.join(example['nl'])
          sql: str = ' '.join([t[1] for t in example['sql']])
          ntid2example[ntid] = {
            'nl': nl,
            'sql': sql
          }
      return ntid2example

    def gen_sql2nl_data(self, output_path: Path, restricted_ntids: Set[str] = None):
      used_ntids: Set[str] = set()
      with open(output_path, 'w') as fout:
        for eid, (ntid, example) in tqdm(enumerate(self.ntid2example.items())):
          if restricted_ntids and ntid not in restricted_ntids:
            continue
          used_ntids.add(ntid)
          sql = example['sql']
          nl = example['nl']
          td = {
            'uuid': f'squall_{eid}',
            'metadata': {
              'sql': sql,
              'nl': nl,
            },
            'table': {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []},
            'context_before': [nl],
            'context_after': []
          }
          fout.write(json.dumps(td) + '\n')
      if restricted_ntids:
        print(f'found sql for {len(used_ntids)} out of {len(restricted_ntids)}')
        print(f'example ids without sql {list(restricted_ntids - used_ntids)[:10]}')

    def get_subset(self, wtq_prep_path: Path, output_path: Path):
      ntids: Set[str] = set()
      with open(wtq_prep_path, 'r') as fin:
        for l in fin:
          ntid = json.loads(l)['uuid']
          assert ntid not in ntids, 'duplicate ntid'
          ntids.add(ntid)
      self.gen_sql2nl_data(output_path, restricted_ntids=ntids)
