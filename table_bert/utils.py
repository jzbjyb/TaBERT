#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import logging
from enum import Enum
import numpy as np
import os


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1


TRANSFORMER_VERSION = None


try:
    if 'USE_TRANSFORMER' in os.environ:
        logging.warning('force to use the new version')
        raise ImportError
    from pytorch_pretrained_bert.modeling import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertConfig,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead, BertLayerNorm, gelu
    )
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`')
except ImportError:
    from transformers.tokenization_bert import BertTokenizer    # noqa
    from transformers.modeling_bert import (    # noqa
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead
    )
    try:
        from transformers.modeling_bert import BertLayerNorm, gelu
    except ImportError:
        from pytorch_pretrained_bert.modeling import BertLayerNorm, gelu
    from transformers.configuration_bert import BertConfig  # noqa

    hf_flag = 'new'
    TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS

# ELECTRA
from transformers import ElectraConfig, ElectraTokenizer, ElectraForMaskedLM, ElectraForPreTraining

# RoBERTa
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM


def compute_mrr(scores: List[float], labels: List[int]):
    return np.mean([1 / (i + 1) for i, (s, r) in enumerate(sorted(zip(scores, labels), key=lambda x: -x[0])) if r])
