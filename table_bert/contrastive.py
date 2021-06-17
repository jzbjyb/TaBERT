import numpy as np
import torch
from torch import nn


class CLIPLoss(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.source_projection = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        self.target_projection = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 1.0), requires_grad=True)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.source_projection, std=self.input_dim ** -0.5)
        nn.init.normal_(self.target_projection, std=self.input_dim ** -0.5)

    def forward(self,
                source,  # (B, emb_size)
                target):  # (B, emb_size)

        # projection
        source = source @ self.source_projection
        target = target @ self.target_projection

        # normalized features
        source = source / source.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_source = logit_scale * source @ target.t()
        logits_per_target = logit_scale * target @ source.t()

        # cross entropy loss
        bs = source.size(0)
        label = torch.arange(bs).to(source.device)
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        pre_source_loss = loss_fct(logits_per_source, label)
        pre_target_loss = loss_fct(logits_per_target, label)
        avg_loss = (pre_source_loss + pre_target_loss) / 2
        return avg_loss
