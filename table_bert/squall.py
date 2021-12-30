from typing import Dict, Set
import json
from tqdm import tqdm
from pathlib import Path
from table_bert.dataset_utils import BasicDataset
from table_bert.wikitablequestions import WikiTQ


class Squall(BasicDataset):
    def __init__(self, json_file: Path, wikitq: WikiTQ = None):
        self.ntid2example = self.load(json_file, wikitq=wikitq)

    @staticmethod
    def load(filepath: Path, wikitq: WikiTQ = None, fromw: str = ' from w') -> Dict[str, Dict]:
      ntid2example: Dict[str, Dict] = {}
      oob = count_fromw = 0
      with open(filepath, 'r') as fin:
        data = json.load(fin)
        for example in data:
          ntid = example['nt']
          # use either WikiTQ or the preprocessed columns in SQUALL
          if wikitq:
            columns = wikitq.get_table(wikitq.wtqid2tableid[ntid])[0]
          else:
            columns = [c[0] for c in example['columns']]
          nl: str = ' '.join(example['nl'])
          sql = []
          for t in example['sql']:
            if t[0] == 'Column':  # match the column index to the corresponding name
              ci = int(t[1].split('_', 1)[0][1:]) - 1
              if ci < len(columns):
                sql.append(columns[ci])
              else:  # TODO: squall annotation error?
                sql.append(t[1])
                oob += 1
            else:
              sql.append(t[1])
          sql = ' '.join(sql)
          has_fromw = sql.find(fromw + ' ') >= 0 or sql.endswith(fromw)
          count_fromw += int(has_fromw)
          sql = sql.replace(fromw + ' ', ' ')
          if sql.endswith(fromw):
            sql = sql[:-len(fromw)]
          ntid2example[ntid] = {
            'nl': nl,
            'sql': sql
          }
      print(f'total: {len(ntid2example)}; column out of bound: {oob}; #examples with "{fromw}": {count_fromw}')
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
              'ntid': ntid,
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
