from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import torch
import faiss


class FaissUtils(object):
    def __init__(self, index_emb_size: int, cuda: bool):
        self.index_emb_size = index_emb_size
        self.cuda = cuda

    def load_span_faiss(self, repr_file: str, index_name: str, query_name: str, normalize: bool = True):
        print('loading ...')
        repr = np.load(repr_file, allow_pickle=True)
        self.index_emb = repr[f'{index_name}_repr'].astype('float32')
        self.index_index = repr[f'{index_name}_index']
        self.index_text = repr[f'{index_name}_text']
        self.query_emb = repr[f'{query_name}_repr'].astype('float32')
        self.query_index = repr[f'{query_name}_index']
        self.query_text = repr[f'{query_name}_text']
        if normalize:
            self.index_emb = self.index_emb / np.sqrt((self.index_emb * self.index_emb).sum(-1, keepdims=True))
            self.query_emb = self.query_emb / np.sqrt((self.query_emb * self.query_emb).sum(-1, keepdims=True))

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
