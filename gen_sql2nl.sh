#!/usr/bin/env bash

source env_initialize.sh

# -- Spider data --
#input_dir=data/spider/train/sqlnl.json
#output_dir=/mnt/root/TaBERT/data/train_data/spider_sql2nl

#input_dir=data/spider/train/sqlnl.json.2048
#output_dir=/mnt/root/TaBERT/data/train_data/spider_sql2nl_num2048

# -- SQUALL data --
#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl/train.src
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl/train.src.1024
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_num1024

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized

input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.4096
output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_num4096

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.128.bm25_top1
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_bm25_top1_num128

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.bm25_top1_wokeyword
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_bm25_top1_wokeyword_num128

# -- SQUALL data (with self training) --
#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num128.beamsearchnl.multitask_filtered.top8192
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_num128_beamsearchnl_multitask_filtered_top8192

# min verification
#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num1024.beamsearchnl.tapex_filtered_min.top8192
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_num1024_beamsearchnl_tapex_filtered_min_top8192

# logprob
#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num1024.beamsearchnl_logprob.tapex_filtered.top8192
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_num1024_beamsearchnl_logprob_tapex_filtered_top8192

# bart as verification
#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num1024.beamsearchnl.bart_filtered.top8192
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_num1024_beamsearchnl_bart_filtered_top8192

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num512.beamsearchnl.tapex_filtered.top8192
#output_dir=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_num512_beamsearchnl_tapex_filtered_top8192

# -- TAPEX data --
#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.sql2nl.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/tapex_05m_sql2nl

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.sql2nl.jsonl.bm25_top1.wtqid
#output_dir=/mnt/root/TaBERT/data/train_data/tapex_05m_sql2nl_bm25_top1

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.sql2nl.jsonl.bm25_top1_wokeyword.wtqid
#output_dir=/mnt/root/TaBERT/data/train_data/tapex_05m_sql2nl_bm25_top1_wokeyword

# --no_shuffle is needed for dev/test
additional_row_count=0
top_row_count=10000
max_source_len=1024
max_target_len=1024
max_context_len=128
max_num_mention_per_example=3
column_delimiter='|'
row_delimiter='none'
mkdir -p ${output_dir}
worldsize=1

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name facebook/bart-base \
    --do_lower_case \
    --epochs_to_generate 1 \
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
    --seq2seq_format sql2nl \
    --table_linearization tapex \
    --dev_num 0 \
    --global_rank $i &
done
wait
