#!/usr/bin/env bash

source env_initialize.sh

output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapas_data0_bart_mlm_contextmention_dense_span_context_as_whole
input_dir=/mnt/root/tapas/data/pretrain/train/preprocessed_mention_dense_span_context_as_whole.jsonl.data0
# --no_shuffle is needed for dev/test
additional_row_count=0
top_row_count=100
max_num_mention_per_example=3
column_delimiter='//'
row_delimiter="[SEP]"
mkdir -p ${output_dir}
worldsize=40

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name facebook/bart-base \
    --do_lower_case \
    --epochs_to_generate 10 \
    --max_context_len 128 \
    --max_column_len 15 \
    --max_cell_len 15 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column | value' \
    --column_delimiter ${column_delimiter} \
    --row_delimiter ${row_delimiter} \
    --world_size ${worldsize} \
    --additional_row_count ${additional_row_count} \
    --top_row_count ${top_row_count} \
    --max_num_mention_per_example ${max_num_mention_per_example} \
    --use_sampled_value \
    --mask_value \
    --mask_value_column_separate \
    --skip_column_name_longer_than 0 \
    --not_skip_empty_column_name \
    --seq2seq_format mlm_mention-context \
    --dev_num 0 \
    --global_rank $i &
done
wait
