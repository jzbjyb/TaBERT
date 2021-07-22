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
    class BertTokenizerWrapper(BertTokenizer):
        def convert_ids_to_tokens(self, ids):
            tokens = []
            for i in ids:
                if i < 0:
                    tokens.append(None)
                else:
                    tokens.append(self.ids_to_tokens[i])
            return tokens

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`')
except ImportError:
    from transformers.tokenization_bert import BertTokenizer    # noqa
    BertTokenizerWrapper = BertTokenizer
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

# BERT
from transformers import BertTokenizerFast

# ELECTRA
from transformers import ElectraConfig, ElectraTokenizer, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining

# RoBERTa
# TODO: set add_prefix_space to True?
from transformers import RobertaConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaForMaskedLM

# BART
from transformers import BartConfig, BartTokenizer, BartTokenizerFast, BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right


def compute_mrr(scores: List[float], labels: List[int]):
    return np.mean([1 / (i + 1) for i, (s, r) in enumerate(sorted(zip(scores, labels), key=lambda x: -x[0])) if r])


class BartTokenizerWrapper(BartTokenizer):
    def tokenize(self, *args, **kwargs):
        if 'add_prefix_space' in kwargs:
            del kwargs['add_prefix_space']
        return super(BartTokenizerWrapper, self).tokenize(*args, **kwargs, add_prefix_space=True)  # always add space

class BartTokenizerFastWrapper(BartTokenizerFast):
    def __call__(self, *args, **kwargs):
        if type(args[0]) is not str:
            raise NotImplementedError
        if len(args[0]) > 0 and args[0][0] != ' ':
            args = (' ' + args[0],) + args[1:]  # add_prefix_space
            result = super(BartTokenizerFastWrapper, self).__call__(*args, **kwargs)
            result['added_prefix_space'] = True
        else:
            result = super(BartTokenizerFastWrapper, self).__call__(*args, **kwargs)
        return result
