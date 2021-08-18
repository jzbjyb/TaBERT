from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import torch
import faiss


class FaissUtils(object):
    def __init__(self, index_emb_size: int, cuda: bool):
        self.index_emb_size = index_emb_size
        self.cuda = cuda

    def convert_index(self, num_shards: int, total_count: int):
        raw_ind = list(range(total_count))
        rearranged_ind = []
        for s in range(num_shards):
            rearranged_ind.extend(raw_ind[s::num_shards])
        to_raw_ind = dict(zip(raw_ind, rearranged_ind))
        self.index_index = np.array([to_raw_ind[i] for i in self.index_index])
        self.query_index = np.array([to_raw_ind[i] for i in self.query_index])

    def load_span_faiss(self, repr_files: List[str], index_name: str, query_name: str, normalize: bool = True, reindex_shards: int = None):
        print('loading ...')
        index_emb_li = []
        index_index_li = []
        index_text_li = []
        query_emb_li = []
        query_index_li = []
        query_text_li = []
        for repr_file in repr_files:
            repr = np.load(repr_file, allow_pickle=True)
            index_emb_li.append(repr[f'{index_name}_repr'].astype('float32'))
            index_index_li.append(repr[f'{index_name}_index'])
            index_text_li.append(repr[f'{index_name}_text'])
            query_emb_li.append(repr[f'{query_name}_repr'].astype('float32'))
            query_index_li.append(repr[f'{query_name}_index'])
            query_text_li.append(repr[f'{query_name}_text'])
        self.index_emb = np.concatenate(index_emb_li)
        del index_emb_li[:]
        self.index_emb_size = self.index_emb.shape[1]
        self.index_index = np.concatenate(index_index_li)
        self.index_index_set = set(self.index_index)
        self.index_text = np.concatenate(index_text_li)
        self.query_emb = np.concatenate(query_emb_li)
        del query_emb_li[:]
        self.query_emb_size = self.query_emb.shape[1]
        self.query_index = np.concatenate(query_index_li)
        self.query_index_set = set(self.query_index)
        self.query_text = np.concatenate(query_text_li)
        if normalize:
            self.index_emb = self.index_emb / np.sqrt((self.index_emb * self.index_emb).sum(-1, keepdims=True))
            self.query_emb = self.query_emb / np.sqrt((self.query_emb * self.query_emb).sum(-1, keepdims=True))
        if reindex_shards:
            total_count = np.max(self.index_index) + 1
            assert total_count == 218419, '3merge data only!'
            self.convert_index(reindex_shards, total_count)

    def build_small_index(self, size: int):
        sample_inds = np.random.choice(self.index_emb.shape[0], size, replace=False)
        self.small_index_emb = self.index_emb[sample_inds]
        self.small_index_index = self.index_index[sample_inds]
        self.small_index_text = self.index_text[sample_inds]
        self.small_index = faiss.IndexHNSWFlat(self.small_index_emb.shape[1], self.index_emb_size, faiss.METRIC_INNER_PRODUCT)
        self.small_index.add(self.small_index_emb)

    def get_text_mask(self, text: List[str]):
        mask = [False if type(t) not in {str, np.str_} or t == '' else True for t in text]
        return mask

    def get_subset_query_emb(self, sub_index: List[int]):
        if len(set(sub_index) & self.query_index_set) <= 0:
            return {
                'emb': np.zeros((0, self.query_emb_size)),
                'index': np.zeros(0).astype(int),
                'text': np.zeros(0).astype(str)
            }
        mask = np.isin(self.query_index, sub_index)
        sub_query_emb = self.query_emb[mask]
        sub_query_index = self.query_index[mask]
        sub_query_text = self.query_text[mask]
        sub_mask = self.get_text_mask(sub_query_text)
        sub_query_emb = sub_query_emb[sub_mask]
        sub_query_index = sub_query_index[sub_mask]
        sub_query_text = sub_query_text[sub_mask]
        return {
            'emb': sub_query_emb,
            'index': sub_query_index,
            'text': sub_query_text
        }

    def query_from_subset(self, query_emb: np.ndarray, sub_index: List[int], topk: int, use_faiss: bool = True):
        if len(set(sub_index) & self.index_index_set) <= 0:
            return [{} for i in range(len(query_emb))]

        # select
        mask = np.isin(self.index_index, sub_index)
        sub_index_emb = self.index_emb[mask]
        sub_index_index = self.index_index[mask]
        sub_index_text = self.index_text[mask]
        sub_mask = self.get_text_mask(sub_index_text)
        sub_index_emb = sub_index_emb[sub_mask]
        sub_index_index = sub_index_index[sub_mask]
        sub_index_text = sub_index_text[sub_mask]

        if use_faiss:
            # index
            index = faiss.IndexHNSWFlat(sub_index_emb.shape[1], self.index_emb_size, faiss.METRIC_INNER_PRODUCT)
            index.add(sub_index_emb)
            # query
            D, I = index.search(query_emb, topk)
        else:
            if self.cuda:
                query_emb = torch.tensor(query_emb).cuda()
                sub_index_emb = torch.tensor(sub_index_emb).cuda()
                D = query_emb @ sub_index_emb.T
                _topk = min(topk, D.shape[1])
                D, I = torch.topk(D, _topk, 1)
                D, I = D.cpu().numpy(), I.cpu().numpy()
            else:
                D = query_emb @ sub_index_emb.T
                I = np.argsort(-D, 1)[:, :topk]
                D = np.take_along_axis(D, I, 1)
        li_retid2textscore: List[Dict[int, List[Tuple[str, float]]]] = []
        for i in range(query_emb.shape[0]):
            li_retid2textscore.append(defaultdict(list))
            for id, score in zip(I[i], D[i]):
                rid = sub_index_index[id]
                text = sub_index_text[id]
                li_retid2textscore[-1][rid].append((text, score))
        return li_retid2textscore


class FaissUtilsMulti(object):
    def __init__(self, index_emb_size: int, cuda: bool, num_index: int):
        self.index_emb_size = index_emb_size
        self.cuda = cuda
        self.num_index = num_index

    def load_span_faiss(self, repr_file: str, index_name: str, query_name: str, normalize: bool = True, reindex_shards: int = None, merge: bool = False):
        if merge:
            self.faiss0 = FaissUtils(self.index_emb_size, self.cuda)
            self.faiss0.load_span_faiss([f'{repr_file}.{i}' for i in range(self.num_index)], index_name=index_name, query_name=query_name, normalize=normalize, reindex_shards=reindex_shards)
            self.num_index = 1
        else:
            for i in range(self.num_index):
                setattr(self, f'faiss{i}', FaissUtils(self.index_emb_size, self.cuda))
                getattr(self, f'faiss{i}').load_span_faiss(
                    [f'{repr_file}.{i}'], index_name=index_name, query_name=query_name, normalize=normalize, reindex_shards=reindex_shards)

    def get_subset_query_emb(self, sub_index: List[int]):
        qds = []
        for i in range(self.num_index):
            fi = getattr(self, f'faiss{i}')
            qds.append(fi.get_subset_query_emb(sub_index))
        combined_qd = {k: np.concatenate([qd[k] for qd in qds], 0) for k in qds[0]}
        return combined_qd

    def query_from_subset(self, *args, **kwargs):
        results = []
        for i in range(self.num_index):
            fi = getattr(self, f'faiss{i}')
            results.append(fi.query_from_subset(*args, **kwargs))
        combined = results[0]
        for result in results[1:]:
            assert len(combined) == len(result)
            for i in range(len(combined)):
                for rid, tss in result[i].items():
                    combined[i][rid].extend(tss)
        return combined
