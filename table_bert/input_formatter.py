#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
from random import choice, shuffle, sample, random, randint
from typing import List, Callable, Dict, Any, Union, Set

from table_bert.utils import BertTokenizer
from table_bert.table_bert import MAX_BERT_INPUT_LENGTH, MAX_TARGET_LENGTH
from table_bert.config import TableBertConfig
from table_bert.dataset import Example
from table_bert.table import Column, Table

trim_count = {'total': 0, 'trim': 0}


class TableBertBertInputFormatter(object):
    def __init__(self, config: TableBertConfig, tokenizer: BertTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, 'vocab'):
            self.vocab_list = list(self.tokenizer.vocab.keys())
        elif hasattr(self.tokenizer, 'get_vocab'):
            self.vocab_list = list(self.tokenizer.get_vocab().keys())
        else:
            raise Exception('cannot find vocab for the tokenizer {}'.format(self.tokenizer))

class TableTooLongError(ValueError):
    pass


class VanillaTableBertInputFormatter(TableBertBertInputFormatter):
    def get_cell_input(
        self,
        column: Column,
        cell_value: List[str],
        token_offset: int = 0,
        cell_input_template: List[str] = None,
    ):
        input = []
        span_map = {
            'first_token': (token_offset, token_offset + 1)
        }
        cell_input_template = cell_input_template or self.config.cell_input_template

        for token in cell_input_template:
            start_token_abs_position = len(input) + token_offset
            if token == 'column':
                name_tokens = column.name_tokens if self.config.max_column_len is None else column.name_tokens[:self.config.max_column_len]
                span_map['column_name'] = (start_token_abs_position,
                                           start_token_abs_position + len(name_tokens))
                input.extend(name_tokens)
            elif token == 'value':
                span_map['value'] = (start_token_abs_position,
                                     start_token_abs_position + len(cell_value))
                input.extend(cell_value)
            elif token == 'type':
                span_map['type'] = (start_token_abs_position,
                                    start_token_abs_position + 1)
                input.append(column.type)
            else:
                span_map.setdefault('other_tokens', []).append(start_token_abs_position)
                input.append(token)

        span_map['whole_span'] = (token_offset, token_offset + len(input))

        return input, span_map

    def get_input(self, context: List[str], table: Table, additional_rows: List[List[Any]] = [], trim_long_table: Union[None, int] = 0, shuffle: bool = False):
        row_data = [
            column.sample_value_tokens
            for column in table.header
        ]

        return self.get_row_input(context, table.header, row_data, additional_rows, trim_long_table=trim_long_table, shuffle=shuffle)

    def _concate_cells(self, header: List, row_data: List, table_tokens_start_idx: int, trim_long_table: int, max_table_token_length: int):
        row_input_tokens = []
        column_token_span_maps = []
        column_start_idx = table_tokens_start_idx

        col_id = 0
        for col_id, column in enumerate(header):
            value_tokens = row_data[col_id]
            truncated_value_tokens = value_tokens[:self.config.max_cell_len]

            column_input_tokens, token_span_map = self.get_cell_input(
                column,
                truncated_value_tokens,
                token_offset=column_start_idx
            )
            column_input_tokens.append(self.config.column_delimiter)

            early_stop = False
            if trim_long_table is not None:
                trim_count['total'] += 1
                if len(row_input_tokens) + len(column_input_tokens) > max_table_token_length:
                    trim_count['trim'] += 1
                    valid_column_input_token_len = max_table_token_length - len(row_input_tokens)
                    column_input_tokens = column_input_tokens[:valid_column_input_token_len]
                    end_index = column_start_idx + len(column_input_tokens)
                    keys_to_delete = []
                    for key in token_span_map:
                        if key in {'column_name', 'type', 'value', 'whole_span'}:
                            span_start_idx, span_end_idx = token_span_map[key]
                            if span_start_idx < end_index < span_end_idx:
                                token_span_map[key] = (span_start_idx, end_index)
                            elif end_index <= span_start_idx:
                                keys_to_delete.append(key)
                        elif key == 'other_tokens':
                            old_positions = token_span_map[key]
                            new_positions = [idx for idx in old_positions if idx < end_index]
                            if not new_positions:
                                keys_to_delete.append(key)

                    for key in keys_to_delete:
                        del token_span_map[key]

                    # nothing left, we just skip this cell and break
                    if len(token_span_map) == 0:
                        break

                    early_stop = True
                elif len(row_input_tokens) + len(column_input_tokens) == max_table_token_length:
                    early_stop = True
            elif len(row_input_tokens) + len(column_input_tokens) > max_table_token_length:
                break

            row_input_tokens.extend(column_input_tokens)
            column_start_idx = column_start_idx + len(column_input_tokens)
            column_token_span_maps.append(token_span_map)

            if early_stop: break

        return row_input_tokens, column_token_span_maps, col_id

    def get_row_input(self, context: List[str], header: List[Column], row_data: List[Any], additional_rows: List[List[Any]] = [], trim_long_table: Union[None, int] = 0, shuffle: bool = False):  # none means not trim
        max_total_len = trim_long_table or MAX_BERT_INPUT_LENGTH
        if self.config.context_first:
            table_tokens_start_idx = len(context) + 2  # account for cls and sep
            # account for cls and sep, and the ending sep
            max_table_token_length = max_total_len - len(context) - 2 - 1
        else:
            table_tokens_start_idx = 1  # account for starting cls
            # account for cls and sep, and the ending sep
            max_table_token_length = max_total_len - len(context) - 2 - 1

        # generate table tokens
        ext_header = header * (len(additional_rows) + 1)
        if shuffle and len(additional_rows) > 0:
            # try to see how many rows can fit into the max length
            _, _, col_id = self._concate_cells(
                header * len(additional_rows), [i for r in additional_rows for i in r], table_tokens_start_idx=table_tokens_start_idx,
                trim_long_table=trim_long_table, max_table_token_length=max_table_token_length)
            num_fit_rows = col_id // len(header)
            additional_rows.insert(randint(0, max(num_fit_rows - 1, 0)), row_data)
            ext_row_data = [i for r in additional_rows for i in r]
        else:
            ext_row_data = row_data + [i for r in additional_rows for i in r]

        row_input_tokens, column_token_span_maps, col_id = self._concate_cells(
            ext_header, ext_row_data, table_tokens_start_idx=table_tokens_start_idx, trim_long_table=trim_long_table, max_table_token_length=max_table_token_length)

        # it is possible that the first cell to too long and cannot fit into `max_table_token_length`
        # we need to discard this sample
        if len(row_input_tokens) == 0:
            raise TableTooLongError()

        if row_input_tokens[-1] == self.config.column_delimiter:
            del row_input_tokens[-1]

        if self.config.context_first:
            sequence = [self.config.cls_token] + context + [self.config.sep_token] + row_input_tokens + [self.config.sep_token]
            # segment_ids = [0] * (len(context) + 2) + [1] * (len(row_input_tokens) + 1)
            segment_a_length = len(context) + 2
            context_span = (0, 1 + len(context))
            # context_token_indices = list(range(0, 1 + len(context)))
        else:
            sequence = [self.config.cls_token] + row_input_tokens + [self.config.sep_token] + context + [self.config.sep_token]
            # segment_ids = [0] * (len(row_input_tokens) + 2) + [1] * (len(context) + 1)
            segment_a_length = len(row_input_tokens) + 2
            context_span = (len(row_input_tokens) + 1, len(row_input_tokens) + 1 + 1 + len(context) + 1)
            # context_token_indices = list(range(len(row_input_tokens) + 1, len(row_input_tokens) + 1 + 1 + len(context) + 1))

        instance = {
            'tokens': sequence,
            #'token_ids': self.tokenizer.convert_tokens_to_ids(sequence),
            'segment_a_length': segment_a_length,
            # 'segment_ids': segment_ids,
            'column_spans': column_token_span_maps,
            'context_length': 1 + len(context),  # beginning cls/sep + input question
            'context_span': context_span,
            # 'context_token_indices': context_token_indices
        }

        return instance

    def get_additional_rows(self, example, sample_num_rows: int, exclude: Set[int]={}):
        additional_rows = []
        if len(example.column_data) <= 0 or sample_num_rows <= 0:  # no data or no sample
            return additional_rows

        # get valid rows
        num_rows = len(example.column_data[0])
        valid_rows = []
        for row_idx in range(num_rows):
            if row_idx in exclude:
                continue
            valid = True
            for col_idx, column in enumerate(example.header):
                val = example.column_data[col_idx][row_idx]
                if val is None or len(val) == 0:
                    valid = False
                    break
            if valid:
                valid_rows.append(row_idx)

        # sample sample_num_rows rows
        sampled_rows = sample(valid_rows, min(len(valid_rows), sample_num_rows))
        additional_rows = [[] for _ in range(len(sampled_rows))]
        for col_idx, column in enumerate(example.header):
            for i, row_idx in enumerate(sampled_rows):
                additional_rows[i].append(self.tokenizer.tokenize(example.column_data[col_idx][row_idx]))
        return additional_rows

    def get_a_row(self, example):
        additional_rows = []
        answer = None
        if 'firstansrow' in self.config.seq2seq_format:  # use the first answer
            answer = example.answers[0]
            ans_coord = example.answer_coordinates[0] if len(example.answer_coordinates) > 0 else None
            exclude = {} if ans_coord is None else {ans_coord[0]}
            if self.config.additional_row_count:
                additional_rows = self.get_additional_rows(example, self.config.additional_row_count, exclude=exclude)
        elif self.config.additional_row_count:
            raise NotImplementedError

        for col_idx, column in enumerate(example.header):
            if 'firstansrow' in self.config.seq2seq_format and ans_coord is not None:
                # use the row that contains the first answer
                # if ans_coord does not exist, go to the following conditions
                sampled_value = example.column_data[col_idx][ans_coord[0]]
            elif self.config.use_sampled_value:
                sampled_value = column.sample_value
            else:
                col_values = example.column_data[col_idx]
                col_values = [val for val in col_values if val is not None and len(val) > 0]
                sampled_value = choice(col_values) if len(col_values) > 0 else ''

            if sampled_value is None or len(sampled_value) <= 0:
                # use a special symbol if this column is empty
                sampled_value = '-'

            # print('chosen value', sampled_value)
            sampled_value_tokens = self.tokenizer.tokenize(sampled_value)
            column.sample_value_tokens = sampled_value_tokens

        return additional_rows, answer

    def get_pretraining_instances_from_example(
        self, example: Example,
        context_sampler: Callable
    ):
        instances = []
        context_iter = context_sampler(
            example, self.config.max_context_len, context_sample_strategy=self.config.context_sample_strategy)

        for context in context_iter:
            # row_num = len(example.column_data)
            # sampled_row_id = choice(list(range(row_num)))

            additional_rows, answer = self.get_a_row(example)

            if self.config.seq2seq_format is None:
                instance = self.create_pretraining_instance(context, example.header, additional_rows)
                instance['source'] = example.source
                instances.append(instance)
            else:
                if 'mlm' in self.config.seq2seq_format:  # added a dummy target which is identical to the masked sequence
                    instance = self.create_pretraining_instance(context, example.header, additional_rows)
                    instance['source'] = example.source
                    instance['target_token_ids'] = instance['token_ids']
                    instances.append(instance)
                if 'single' in self.config.seq2seq_format:
                    instances.extend(self.create_seq2seq_instances(context, example.header))
                if 'qa' in self.config.seq2seq_format:
                    instances.extend(self.create_qa_instances(context, example.header, answer, additional_rows))

            if 'qa' in self.config.seq2seq_format:  # for qa format, do not iterative over context (i.e., question)
                break

        return instances

    def _add_single_head(self, raw_tokens: List[str], toadd: List[str], target: List[str], seq_a_len: int):
        need_to_remove = len(raw_tokens) + len(toadd) + 1 - MAX_BERT_INPUT_LENGTH  # sep token
        if need_to_remove > 0:  # overflow
            assert len(raw_tokens) >= need_to_remove + 2, 'the raw input is too short to be removed'  # sep and cls token
            tokens = raw_tokens[:-(need_to_remove + 1)]  # also remove the last sep token
            if tokens[-1] != self.config.sep_token:
                tokens.append(self.config.sep_token)
            tokens = tokens + toadd + [self.config.sep_token]
        else:
            tokens = raw_tokens + toadd + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return instance

    def create_seq2seq_instances(self, context, header: List[Column]):
        instances = []
        seq2seqf = self.config.seq2seq_format
        for hi, single_head in enumerate(header):
            if not single_head.used and not single_head.value_used:
                continue
            if len(header) > 1:  # has remain headers
                remain_table = Table('fake_table', header[:hi] + header[hi + 1:])
                input_instance = self.get_input(context, remain_table)
                raw_tokens = input_instance['tokens']
                seq_a_len = input_instance['segment_a_length']
            else:
                raw_tokens = [self.config.cls_token] + context + [self.config.sep_token]
                seq_a_len = len(context) + 2
            if 'single-c2v' in seq2seqf and single_head.value_used:
                cell_value = single_head.sample_value_tokens[:self.config.max_cell_len]
                target_tokens = [self.config.cls_token] + cell_value[:MAX_TARGET_LENGTH - 2] + [self.config.sep_token]
                ctokens = self.get_cell_input(single_head, cell_value, cell_input_template=['column', '|', 'type', '|'])[0]
                instances.append(self._add_single_head(raw_tokens, ctokens, target_tokens, seq_a_len))
            if 'single-v2c' in seq2seqf and single_head.used:
                cell_value = single_head.sample_value_tokens[:self.config.max_cell_len]
                vtokens = self.get_cell_input(single_head, cell_value, cell_input_template=['|', 'value'])[0]
                ctokens = self.get_cell_input(single_head, cell_value, cell_input_template=['column', '|', 'type'])[0]
                target_tokens = [self.config.cls_token] + ctokens[:MAX_TARGET_LENGTH - 2] + [self.config.sep_token]
                instances.append(self._add_single_head(raw_tokens, vtokens, target_tokens, seq_a_len))
        return instances

    def create_qa_instances(self, context, header: List[Column], answer: str, additional_rows: List[List[Any]] = []):
        # context + table without mask
        table = Table('fake_table', header)
        instance = self.get_input(context, table, additional_rows, shuffle=True)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        # answer as target
        target = [self.config.cls_token] + self.tokenizer.tokenize(answer)[:MAX_TARGET_LENGTH - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_pretraining_instance(self, context, header, additional_rows: List[List[Any]] = []):
        table = Table('fake_table', header)
        input_instance = self.get_input(context, table, additional_rows)  # core format function
        column_spans = input_instance['column_spans']

        column_candidate_indices = [
            (
                list(range(*span['column_name']) if 'column_name' in span else []) +
                list(range(*span['type']) if 'type' in span else []) +
                (
                    span['other_tokens']
                    if random() < 0.01 and 'other_tokens' in span
                    else []
                )
            )
            for col_id, span
            in enumerate(column_spans[:len(header)])  # only mask the first row and used columns if specified
            if not self.config.mask_used_column_prob or header[col_id].used
        ]
        if self.config.mask_value:
            column_candidate_indices_value = [
                list(range(*span['value']) if 'value' in span else [])
                for col_id, span
                in enumerate(column_spans[:len(header)])  # only mask the first row and used columns if specified
                if not self.config.mask_used_column_prob or header[col_id].used
            ]
            assert len(column_candidate_indices_value) == len(column_candidate_indices), 'candidate indices length inconsistent'
            if self.config.mask_value_column_separate:
                column_candidate_indices = column_candidate_indices + column_candidate_indices_value
            else:
                column_candidate_indices = [ci + civ for ci, civ in zip(column_candidate_indices, column_candidate_indices_value)]

        masked_column_prob = None
        if self.config.mask_used_column_prob:
            masked_column_prob = min(self.config.masked_column_prob * len(column_spans[:len(header)]) / (len(column_candidate_indices) or 1), 1.0)

        context_candidate_indices = (
            list(range(*input_instance['context_span']))[1:]
            if self.config.context_first else
            list(range(*input_instance['context_span']))[:-1]
        )

        masked_sequence, masked_lm_positions, masked_lm_labels, info = self.create_masked_lm_predictions(
            input_instance['tokens'], context_candidate_indices, column_candidate_indices, masked_column_prob=masked_column_prob
        )

        info['num_columns'] = len(header)

        instance = {
            "tokens": masked_sequence,
            "token_ids": self.tokenizer.convert_tokens_to_ids(masked_sequence),
            "segment_a_length": input_instance['segment_a_length'],
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_lm_label_ids": self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            "info": info
        }

        return instance

    def create_masked_lm_predictions(
        self,
        tokens, context_indices, column_indices, masked_column_prob=None
    ):
        table_mask_strategy = self.config.table_mask_strategy
        masked_column_prob = masked_column_prob or self.config.masked_column_prob

        info = dict()
        info['num_maskable_column_tokens'] = sum(len(token_ids) for token_ids in column_indices)

        if table_mask_strategy == 'column_token':
            column_indices = [i for l in column_indices for i in l]
            num_column_tokens_to_mask = min(self.config.max_predictions_per_seq,
                                            max(2, int(len(column_indices) * masked_column_prob)))
            shuffle(column_indices)
            masked_column_token_indices = sorted(sample(column_indices, num_column_tokens_to_mask))
        elif table_mask_strategy == 'column':
            num_maskable_columns = len(column_indices)
            num_column_to_mask = max(1, ceil(num_maskable_columns * masked_column_prob))
            columns_to_mask = sorted(sample(list(range(num_maskable_columns)), num_column_to_mask))
            shuffle(columns_to_mask)
            num_column_tokens_to_mask = sum(len(column_indices[i]) for i in columns_to_mask)
            masked_column_token_indices = [idx for col in columns_to_mask for idx in column_indices[col]]

            info['num_masked_columns'] = num_column_to_mask
        else:
            raise RuntimeError('unknown mode!')

        max_context_token_to_mask = self.config.max_predictions_per_seq - num_column_tokens_to_mask
        num_context_tokens_to_mask = min(max_context_token_to_mask,
                                         max(1, int(len(context_indices) * self.config.masked_context_prob)))

        if num_context_tokens_to_mask > 0:
            # if num_context_tokens_to_mask < 0 or num_context_tokens_to_mask > len(context_indices):
            #     for col_id in columns_to_mask:
            #         print([tokens[i] for i in column_indices[col_id]])
            #     print(num_context_tokens_to_mask, num_column_tokens_to_mask)
            shuffle(context_indices)
            masked_context_token_indices = sorted(sample(context_indices, num_context_tokens_to_mask))
            masked_indices = sorted(masked_context_token_indices + masked_column_token_indices)
        else:
            masked_indices = masked_column_token_indices
        assert len(set(masked_column_token_indices)) == len(masked_column_token_indices), 'duplicate indicies'

        masked_token_labels = []

        for index in masked_indices:
            if not self.config.use_electra:  # BERT style masking
                # 80% of the time, replace with mask
                if random() < 0.8:
                    masked_token = self.config.mask_token
                else:
                    # 10% of the time, keep original
                    if random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = choice(self.vocab_list)
            else:  # ELECTRA style masking
                if random() < 0.85:  # 85% of the time, replace with mask
                    masked_token = self.config.mask_token
                else:  # 15% of the time, keep original
                    masked_token = tokens[index]
            masked_token_labels.append(tokens[index])
            # Once we've saved the true label for that token, we can overwrite it with the masked version
            tokens[index] = masked_token

        info.update({
            'num_column_tokens_to_mask': num_column_tokens_to_mask,
            'num_context_tokens_to_mask': num_context_tokens_to_mask,
        })

        return tokens, masked_indices, masked_token_labels, info

    def remove_unecessary_instance_entries(self, instance: Dict):
        del instance['tokens']
        del instance['masked_lm_labels']
        del instance['info']


if __name__ == '__main__':
    config = TableBertConfig()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_formatter = VanillaTableBertInputFormatter(config, tokenizer)

    header = []
    for i in range(1000):
        header.append(
            Column(
                name='test',
                type='text',
                name_tokens=['test'] * 3,
                sample_value='ha ha ha yay',
                sample_value_tokens=['ha', 'ha', 'ha', 'yay']
            )
        )

    print(
        input_formatter.get_row_input(
            context='12 213 5 345 23 234'.split(),
            header=header,
            row_data=[col.sample_value_tokens for col in header]
        )
    )
