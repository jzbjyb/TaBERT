#!/usr/bin/env bash

source env_initialize.sh

# -- TAPEX data --
#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_withtable_denormalized.preprocessed.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_withtable_denormalized_qa

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_withtable_denormalized.num16_preprocessed.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_withtable_denormalized_num16_qa

input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.num4096_preprocessed.jsonl
output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_num4096_qa

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.preprocessed.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_qa

# -- WTQ data (for self training candidate evaluation) --
#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num128.samplenl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_wtq_wtqnl_denormalized_num128_sample

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num1024.beamsearchnl_logprob
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_wtq_wtqnl_denormalized_num1024_beamsearch_logprob

#input_dir=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num512.beamsearchnl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_wtq_wtqnl_denormalized_num512_beamsearch

# -- TAPEX data (after self training) --
#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.num128_preprocessed.beamsearchnl_multitask_filtered_top8192_mt.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_num128_qa_beamsearchnl_multitask_filtered_top8192_mt

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.num1024_preprocessed.beamsearchnl_tapex_filtered_min_top8192_mt.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_num1024_qa_beamsearchnl_tapex_filtered_min_top8192_mt

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.num1024_preprocessed.beamsearchnl_logprob_tapex_filtered_top8192_mt.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_num1024_qa_beamsearchnl_logprob_tapex_filtered_top8192_mt

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.num1024_preprocessed.beamsearchnl_bart_filtered_top8192_mt.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_num1024_qa_beamsearchnl_bart_filtered_top8192_mt

#input_dir=/mnt/root/TaBERT/data/tapex/preprocessed/train.500k.wtq_nl_denormalized.num512_preprocessed.beamsearchnl_tapex_filtered_top8192_mt.jsonl
#output_dir=/mnt/root/TaBERT/data/train_data/wholetable_tapex_wtqnl_denormalized_num512_qa_beamsearchnl_tapex_filtered_top8192_mt

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
    --seq2seq_format qa_tapex \
    --table_linearization tapex \
    --skip_sep_in_middle \
    --dev_num 0 \
    --global_rank $i &
done
wait
