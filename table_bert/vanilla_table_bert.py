#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import sys
from typing import List, Any, Tuple, Dict
import numpy as np
from fairseq import distributed_utils
from tqdm import tqdm
import json
import torch
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_max, scatter_mean

from table_bert.utils import BertForPreTraining, BertForMaskedLM, TRANSFORMER_VERSION, TransformerVersion
from table_bert.utils import ElectraForPreTraining, ElectraForMaskedLM, RobertaForMaskedLM, \
    BartForConditionalGeneration, shift_tokens_right
from table_bert.table_bert import TableBertModel
from table_bert.config import TableBertConfig, BERT_CONFIGS
from table_bert.table import Table
from table_bert.input_formatter import VanillaTableBertInputFormatter
from table_bert.electra import ELECTRAModel, ELECTRALoss
from table_bert.contrastive import CLIPLoss


class VanillaTableBert(TableBertModel):
    CONFIG_CLASS = TableBertConfig

    def __init__(
        self,
        config: TableBertConfig,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        getattr(self, 'load_{}'.format(config.model_type.value))()  # load model based on model type
        if config.load_model_from is not None:
            print('init from {}'.format(config.load_model_from))
            state_dict = torch.load(config.load_model_from, map_location='cpu')
            self.load_state_dict(state_dict, strict=True)
        obj = self.config.objective_function
        if 'contrastive' in obj:
            self.contrastive_loss = CLIPLoss(self.output_size, self.config.contrastive_emb_size)
        elif 'contrast-concat' in obj:
            self.contrastive_loss = CLIPLoss(self.output_size, self.config.contrastive_emb_size, is_paired=True)
        elif 'nsp' in obj or 'binary' in obj:
            self.nsp_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.input_formatter = VanillaTableBertInputFormatter(self.config, self.tokenizer)

    def load_bert(self):
        obj = self.config.objective_function
        for loss_fct in ['seq2seq']:
            if loss_fct in obj:
                raise NotImplementedError
        if 'nsp' in obj or 'binary' in obj:
            self._bert_model = BertForPreTraining.from_pretrained(self.config.base_model_name)
        else:
            self._bert_model = BertForMaskedLM.from_pretrained(self.config.base_model_name)

    def load_electra(self):
        for loss_fct in ['nsp', 'binary', 'seq2seq']:
            if loss_fct in self.config.objective_function:
                raise NotImplementedError
        generator = ElectraForMaskedLM.from_pretrained(self.config.base_model_name)
        discriminator = ElectraForPreTraining.from_pretrained(
            self.config.base_model_name.replace('generator', 'discriminator'))
        discriminator.electra.embeddings = generator.electra.embeddings
        generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight
        self._electra = ELECTRAModel(generator, discriminator)
        self._electra_loss = ELECTRALoss()

    def load_roberta(self):
        for loss_fct in ['nsp', 'binary', 'seq2seq']:
            if loss_fct in self.config.objective_function:
                raise NotImplementedError
        self._roberta = RobertaForMaskedLM.from_pretrained(self.config.base_model_name)

    def load_bart(self):
        for loss_fct in ['nsp', 'binary', 'contrastive', 'contrast-concat']:
            if loss_fct in self.config.objective_function:
                raise NotImplementedError
        self._bart = BartForConditionalGeneration.from_pretrained(self.config.base_model_name)

    def forward(self, *args, **kwargs):
        return getattr(self, 'forward_{}'.format(self.config.model_type.value))(*args, **kwargs)

    def forward_bert(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = masked_lm_labels.ne(-1).sum().item()
        obj = self.config.objective_function
        if 'mlm' in obj or 'binary' in obj:
            sequence_output, pooled_output = self._bert_model.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            if type(self._bert_model) is BertForMaskedLM:
                prediction_scores = self._bert_model.cls(sequence_output)
            elif type(self._bert_model) is BertForPreTraining:
                prediction_scores, seq_relationship_score = self._bert_model.cls(sequence_output, pooled_output)
            if 'mlm' in obj:
                loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
                masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), masked_lm_labels.view(-1))
                total_loss += masked_lm_loss
            if 'binary' in obj:
                binary_label = 1 - kwargs['is_positives']  # 0 => next sentence is the continuation, 1 => next sentence is a random sentence
                binary_loss = self.nsp_loss(seq_relationship_score.view(-1, 2), binary_label.view(-1))
                total_loss += binary_loss
        if 'contrastive' in obj:
            # use the representation corresponding to the first token (cls or sep)
            context_repr = self._bert_model.bert(
                kwargs['context_input_ids'], kwargs['context_token_type_ids'], kwargs['context_attention_mask'],
                output_all_encoded_layers=False)[0][:, 0, :]
            table_repr = self._bert_model.bert(
                kwargs['table_input_ids'], kwargs['table_token_type_ids'], kwargs['table_attention_mask'],
                output_all_encoded_layers=False)[0][:, 0, :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr)
            total_loss += contrastive_loss
        if 'contrast-concat' in obj:
            # use the representation corresponding to the first token (cls or sep)
            concat_repr, _ = self._bert_model.bert(
                kwargs['concat_input_ids'], kwargs['concat_token_type_ids'], kwargs['concat_attention_mask'], output_all_encoded_layers=False)
            context_repr = concat_repr[kwargs['context_mask'], :]
            table_repr = concat_repr[kwargs['table_mask'], :]
            l, binary_label = CLIPLoss.get_diag_label(context_repr, binary=True)
            contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=binary_label)
            total_loss += contrastive_loss
        if 'nsp' in obj:
            # only use cls
            sequence_output, pooled_output = self._bert_model.bert(
                kwargs['concat_input_ids'], kwargs['concat_token_type_ids'], kwargs['concat_attention_mask'], output_all_encoded_layers=False)
            _, seq_relationship_score = self._bert_model.cls(sequence_output, pooled_output)
            l, nsp_label = CLIPLoss.get_diag_label(pooled_output, binary=True)
            nsp_label = 1 - nsp_label  # 0 => next sentence is the continuation, 1 => next sentence is a random sentence
            nsp_loss = self.nsp_loss(seq_relationship_score.view(-1, 2), nsp_label.view(-1))
            total_loss += nsp_loss

        logging_output = {
            'sample_size': sample_size,
            'loss': total_loss.item()
        }
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def evaluate_bert(self, batch):
        results: List[Dict] = []
        input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        repr = self._bert_model.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)[0]
        logits = self._bert_model.cls(repr)
        pred_ids = logits.max(-1)[1]
        gold_ids = batch['masked_lm_labels']
        for tgt_id, pred_id, gold_id in zip(input_ids, pred_ids, gold_ids):
            tgt = self.tokenizer.convert_ids_to_tokens(tgt_id.detach().cpu().numpy())
            pred = self.tokenizer.convert_ids_to_tokens(pred_id.detach().cpu().numpy())
            gold = self.tokenizer.convert_ids_to_tokens(gold_id.detach().cpu().numpy())
            result = {'tgt': tgt, 'pred': pred, 'gold': gold}
            results.append(result)
        return results

    def forward_electra(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = 0
        if 'mlm' in self.config.objective_function:
            gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied = \
                self._electra(input_ids, token_type_ids, attention_mask, masked_lm_labels)
            electra_loss = self._electra_loss(
                (gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied), masked_lm_labels)
            sample_size = masked_lm_labels.ne(-1).sum().item()
            total_loss += electra_loss
        if 'contrastive' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            context_repr = self._electra.discriminator.electra(
                kwargs['context_input_ids'],  kwargs['context_attention_mask'], kwargs['context_token_type_ids'])[0][:, 0, :]
            table_repr = self._electra.discriminator.electra(
                kwargs['table_input_ids'], kwargs['table_attention_mask'], kwargs['table_token_type_ids'])[0][:, 0, :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr)
            total_loss += contrastive_loss
        if 'contrast-concat' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            concat_repr = self._electra.discriminator.electra(
                kwargs['concat_input_ids'], kwargs['concat_attention_mask'], kwargs['concat_token_type_ids'])[0]
            context_repr = concat_repr[kwargs['context_mask'], :]
            table_repr = concat_repr[kwargs['table_mask'], :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=None)
            total_loss += contrastive_loss

        logging_output = {
            'sample_size': sample_size,
            'loss': total_loss.item()
        }
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def forward_roberta(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = 0
        if 'mlm' in self.config.objective_function:
            sequence_logits = self._roberta(input_ids, attention_mask)[0]
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            masked_lm_loss = loss_fct(sequence_logits.view(-1, sequence_logits.size(-1)), masked_lm_labels.view(-1))
            sample_size = masked_lm_labels.ne(-1).sum().item()
            total_loss += masked_lm_loss
        if 'contrastive' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            context_repr = self._roberta.roberta(kwargs['context_input_ids'], kwargs['context_attention_mask'])[0][:, 0, :]
            table_repr = self._roberta.roberta(kwargs['table_input_ids'], kwargs['table_attention_mask'])[0][:, 0, :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr)
            total_loss += contrastive_loss
        if 'contrast-concat' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            concat_repr = self._roberta.roberta(kwargs['concat_input_ids'], kwargs['concat_attention_mask'])[0]
            context_repr = concat_repr[kwargs['context_mask'], :]
            table_repr = concat_repr[kwargs['table_mask'], :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=None)
            total_loss += contrastive_loss

        logging_output = {
            'sample_size': sample_size,
            'loss': total_loss.item()
        }
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def forward_bart(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = masked_lm_labels.ne(-1).sum().item()
        logging_output = {'sample_size': sample_size}

        if 'mlm' in self.config.objective_function:
            sequence_logits = self._bart(input_ids, attention_mask=attention_mask, return_dict=True).logits
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            masked_lm_loss = loss_fct(sequence_logits.view(-1, sequence_logits.size(-1)), masked_lm_labels.view(-1))
            total_loss += masked_lm_loss

        for obj in ['text2table', 'table2text']:
            if obj not in self.config.objective_function:
                continue
            # get src, tgt
            if obj == 'text2table':
                src_ids, src_mask = kwargs['context_input_ids'], kwargs['context_attention_mask']
                tgt_ids = kwargs['table_input_ids']
            elif obj == 'table2text':
                src_ids, src_mask = kwargs['table_input_ids'], kwargs['table_attention_mask']
                tgt_ids = kwargs['context_input_ids']
            # compute loss
            decoder_input_ids = shift_tokens_right(tgt_ids, self.config.pad_id)
            logits = self._bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_id, reduction='mean')
            seq2seq_loss = loss_fct(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
            total_loss += seq2seq_loss

        if 'seq2seq' in self.config.objective_function:
            src_ids, src_mask = input_ids, attention_mask
            tgt_ids = kwargs['target_input_ids']
            bs = src_ids.size(0)
            decoder_input_ids = shift_tokens_right(tgt_ids, self.config.pad_id)
            logits = self._bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits

            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            # masked_lm_labels is a combination of real mlm and seq2seq targets
            combined_loss = loss_fct(logits.view(-1, logits.size(-1)), masked_lm_labels.view(-1)).view(bs, -1)  # (bs, seq_len)
            combined_loss_avg = combined_loss.sum() / (masked_lm_labels.ne(-1).sum() or 1.0)
            total_loss += combined_loss_avg  # loss is proportional to the number of tokens of each type (mlm or seq2seq)
            # separate mlm and seq2seq loss for logging
            is_mlm = kwargs['is_mlm']  # (bs, )
            mlm_loss_avg = combined_loss[is_mlm].sum() / (masked_lm_labels[is_mlm].ne(-1).sum() or 1.0)
            seq2seq_loss_avg = combined_loss[~is_mlm].sum() / (masked_lm_labels[~is_mlm].ne(-1).sum() or 1.0)
            logging_output['mlm_loss'] = mlm_loss_avg.item()
            logging_output['seq2seq_loss'] = seq2seq_loss_avg.item()

        logging_output['loss'] = total_loss.item()
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def evaluate_bart(self, batch):
        results: List[Dict] = []
        src_ids, src_mask = batch['input_ids'], batch['attention_mask']
        tgt_ids = batch['target_input_ids']
        decoder_input_ids = shift_tokens_right(tgt_ids, self.config.pad_id)
        logits = self._bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids,
                            use_cache=False, return_dict=True).logits
        pred_ids = logits.max(-1)[1]
        gold_ids = batch['masked_lm_labels']
        for tgt_id, pred_id, gold_id in zip(tgt_ids, pred_ids, gold_ids):
            tgt = self.tokenizer.convert_ids_to_tokens(tgt_id)
            pred = self.tokenizer.convert_ids_to_tokens(pred_id)
            gold = self.tokenizer.convert_ids_to_tokens(gold_id)
            result = {'tgt': tgt, 'pred': pred, 'gold': gold}
            results.append(result)
        return results

    def encode_context_and_table(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_token_indices: torch.Tensor,
        context_token_mask: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_mask: torch.Tensor,
        return_bert_encoding: bool = False,
        **kwargs
    ):

        # print('input_ids', input_ids.size(), file=sys.stderr)
        # print('segment_ids', segment_ids.size(), file=sys.stderr)
        # print('attention_mask', attention_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.sum(dim=-1), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.size(), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.sum(dim=-1), file=sys.stderr)
        # print('column_mask', column_mask.size(), file=sys.stderr)

        kwargs = (
            {}
            if TRANSFORMER_VERSION == TransformerVersion.TRANSFORMERS
            else {'output_all_encoded_layers': False}
        )
        sequence_output, _ = self.bert(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask,
            **kwargs
        )
        # except:
        #     print('!!!!!Exception!!!!!')
        #     datum = (input_ids, segment_ids, attention_mask, question_token_mask,
        #              column_token_mask, column_token_to_column_id, column_mask)
        #     torch.save(datum, 'debug.tensors.bin')
        #     raise

        # gather column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        # (batch_size, max_column_size, encoding_size)
        column_encoding = self.get_column_representation(
            flattened_column_encoding,
            column_token_to_column_id,
            column_token_mask,
            column_mask,
            aggregator=self.config.column_representation
        )

        # (batch_size, context_len, encoding_size)
        context_encoding = torch.gather(
            sequence_output,
            dim=1,
            index=context_token_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
        )
        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        encoding_info = {}
        if return_bert_encoding:
            encoding_info['bert_encoding'] = sequence_output

        return context_encoding, column_encoding, encoding_info

    @staticmethod
    def get_column_representation(
        flattened_column_encoding: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_mask: torch.Tensor,
        aggregator: str = 'mean_pool'
    ) -> torch.Tensor:
        """
        Args:
            flattened_column_encoding: (batch_size, total_column_token_num, encoding_size)
            column_token_to_column_id: (batch_size, total_column_token_num + 1)
            column_mask: (batch_size, max_column_num)
            aggregator: ['mean_pool', 'max_pool', 'first_token']
        Returns:
            column_encoding: (batch_size, max_column_num, encoding_size)
        """

        if aggregator.startswith('max_pool'):
            agg_func = scatter_max
            flattened_column_encoding[column_token_mask == 0] = float('-inf')
        elif aggregator.startswith('mean_pool') or aggregator.startswith('first_token'):
            agg_func = scatter_mean
        else:
            raise ValueError(f'Unknown column representation method {aggregator}')

        max_column_num = column_mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size + 1, encoding_size)
        result = agg_func(flattened_column_encoding,
                          column_token_to_column_id.unsqueeze(-1).expand(-1, -1, flattened_column_encoding.size(-1)),
                          dim=1,
                          dim_size=max_column_num + 1)

        # remove the last "garbage collection" entry, mask out padding columns
        result = result[:, :-1] * column_mask.unsqueeze(-1)

        if aggregator == 'max_pool':
            column_encoding = result[0]
        else:
            column_encoding = result

        return column_encoding

    def to_tensor_dict(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        table_specific_tensors=True
    ):
        instances = []
        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            instance = self.input_formatter.get_input(context, table)
            instances.append(instance)

        batch_size = len(contexts)
        max_sequence_len = max(len(x['tokens']) for x in instances)

        # basic tensors
        input_array = np.zeros((batch_size, max_sequence_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

        # table specific tensors
        if table_specific_tensors:
            max_column_num = max(len(x['column_spans']) for x in instances)
            max_context_len = max(x['context_length'] for x in instances)

            context_token_indices = np.zeros((batch_size, max_context_len), dtype=np.int)
            context_mask = np.zeros((batch_size, max_context_len), dtype=np.bool)
            column_token_mask = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

            # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
            column_token_to_column_id = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            column_token_to_column_id.fill(max_column_num)

            column_mask = np.zeros((batch_size, max_column_num), dtype=np.bool)

            column_span = 'whole_span'
            if 'column_name' in self.config.column_representation:
                column_span = 'column_name'
            elif 'first_token' in self.config.column_representation:
                column_span = 'first_token'

        for i, instance in enumerate(instances):
            token_ids = self.tokenizer.convert_tokens_to_ids(instance['tokens'])

            input_array[i, :len(token_ids)] = token_ids
            segment_array[i, instance['segment_a_length']: len(token_ids)] = 1
            mask_array[i, :len(token_ids)] = 1.

            if table_specific_tensors:
                context_token_indices[i, :instance['context_length']] = list(range(*instance['context_span'])) #instance['context_token_indices']
                context_mask[i, :instance['context_length']] = 1.

                header = tables[i].header
                for col_id, column in enumerate(header):
                    if col_id < len(instance['column_spans']):
                        col_start, col_end = instance['column_spans'][col_id][column_span]

                        column_token_to_column_id[i, col_start: col_end] = col_id
                        column_token_mask[i, col_start: col_end] = 1.
                        column_mask[i, col_id] = 1.

        tensor_dict = {
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'segment_ids': torch.tensor(segment_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array, dtype=torch.float32),
        }

        if table_specific_tensors:
            tensor_dict.update({
                'context_token_indices': torch.tensor(context_token_indices.astype(np.int64)),
                'context_token_mask': torch.tensor(context_mask, dtype=torch.float32),
                'column_token_to_column_id': torch.tensor(column_token_to_column_id.astype(np.int64)),
                'column_token_mask': torch.tensor(column_token_mask, dtype=torch.float32),
                'column_mask': torch.tensor(column_mask, dtype=torch.float32)
            })

        # for instance in instances:
        #     print(instance)

        return tensor_dict, instances

    def encode(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        return_bert_encoding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        device = next(self.parameters()).device

        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(device)

        context_encoding, column_encoding, encoding_info = self.encode_context_and_table(
            **tensor_dict,
            return_bert_encoding=return_bert_encoding
        )

        info = {
            'tensor_dict': tensor_dict,
            'instances': instances,
            **encoding_info
        }

        return context_encoding, column_encoding, info

    def validate(self, data_loader, args):
        was_training = self.training
        self.eval()

        logging_info_list = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc=f"Evaluation", file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    _, logging_info = self(**batch)
                    logging_info_list.append(logging_info)
                    pbar.update(1)

        if was_training:
            self.train()

        stats = {
            k: np.average([x[k] for x in logging_info_list])
            for k in logging_info_list[0]
        } if len(logging_info_list) > 0 else {}

        # handel distributed evaluation
        if args.multi_gpu:
            stats = distributed_utils.all_gather_list(stats)
            stats = {
                k: np.average([x[k] for x in stats])
                for k in stats[0]
            } if len(stats) > 0 else {}

        return stats

    def evaluate(self, data_loader, args):
        output_file = 'evaluation.jsonl' if args.output_file is None else args.output_file
        was_training = self.training
        self.eval()

        with torch.no_grad(), open(args.output_dir / output_file, 'w') as fout:
            with tqdm(total=len(data_loader), desc='Evaluation', file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    for result in getattr(self, f'evaluate_{self.config.model_type.value}')(batch):
                        fout.write(json.dumps(result) + '\n')
                    pbar.update(1)

        if was_training:
            self.train()

    def generate(self, data_loader, args):
        output_file = 'generation.tsv' if args.output_file is None else args.output_file
        was_training = self.training
        self.eval()

        results: List[str] = []
        with torch.no_grad(), open(args.output_dir / output_file, 'w') as fout:
            with tqdm(total=len(data_loader), desc='Generation', file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    target_ids = self._bart.generate(
                        batch['input_ids'], attention_mask=batch['attention_mask'],
                        num_beams=args.num_beams, min_length=args.min_generate_length,
                        max_length=args.max_generate_length, early_stopping=True)
                    gold_ids = batch['target_input_ids']
                    for input_id, target_id, gold_id in zip(batch['input_ids'], target_ids, gold_ids):
                        source = self.tokenizer.decode(input_id, skip_special_tokens=False).replace('\n', '\\n')
                        pred = self.tokenizer.decode(target_id, skip_special_tokens=False).replace('\n', '\\n')
                        gold = self.tokenizer.decode(gold_id, skip_special_tokens=False).replace('\n', '\\n')
                        results.append(pred)
                        fout.write(f'{pred}\t{gold}\t{source}\n')
                    pbar.update(1)

        if was_training:
            self.train()
