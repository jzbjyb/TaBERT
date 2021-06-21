import torch
import torch.nn as nn
from table_bert import TableBertModel


class Ranker(nn.Module):
    def __init__(self, model: TableBertModel):
        nn.Module.__init__(self)
        self.model = model
        self.tokenizer = model.tokenizer


    def forward(self, questions, tables):
        question_repr, column_repr, _ = self.model.encode(contexts=questions, tables=tables)
        question_repr = question_repr.mean(1)
        column_repr = column_repr.mean(1)
        question_repr = question_repr / question_repr.norm(dim=-1, keepdim=True)
        column_repr = column_repr / column_repr.norm(dim=-1, keepdim=True)
        score = (question_repr * column_repr).sum(-1)
        return score
