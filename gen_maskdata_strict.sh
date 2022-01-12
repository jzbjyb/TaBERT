#!/usr/bin/env bash

source env_initialize.sh

# -- TAPAS data --
#input_dir=/mnt/root/tapas/data/pretrain/train/preprocessed_mention_samepage_bm25.jsonl.data
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapas_samepag_bm25_bartmask_salientmask

#input_dir=/mnt/root/tapas/data/pretrain/train/preprocessed_mention_samepage_dense_bartlarge.06.data0
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapas_samepage_dense_bartlarge_06_bartmask_salientmask

input_dir=/mnt/root/tapas/data/pretrain/train/preprocessed_mention_samepage_dense_bartlarge.06.data0
output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapas_samepage_dense_bartlarge_06_salientmask  # TODO: change seq2seq_format

# --no_shuffle is needed for dev/test
additional_row_count=0
top_row_count=10000
max_source_len=1024
max_target_len=1024
max_context_len=128
max_num_mention_per_example=3
column_delimiter='|'
column_delimiter_first=':'
row_delimiter='none'
mkdir -p ${output_dir}
worldsize=20

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name facebook/bart-base \
    --do_lower_case \
    --epochs_to_generate 10 \
    --max_source_len ${max_source_len} \
    --max_target_len ${max_target_len} \
    --max_context_len ${max_context_len} \
    --max_column_len 15 \
    --max_cell_len 15 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'value' \
    --column_delimiter ${column_delimiter} \
    --column_delimiter_first ${column_delimiter_first} \
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
    --seq2seq_format salient-mask \
    --table_linearization tapex \
    --skip_sep_in_middle \
    --dev_num 0 \
    --global_rank $i &
done
wait
