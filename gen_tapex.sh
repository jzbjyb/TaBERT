#!/usr/bin/env bash

source env_initialize.sh

input_dir=/mnt/root/TaBERT/data/tapex/raw_data/train.src
output_dir=/mnt/root/TaBERT/data/train_data/tapex_bart
# --no_shuffle is needed for dev/test
max_source_len=1024
max_target_len=1024
max_context_len=128
mkdir -p ${output_dir}
worldsize=40

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name facebook/bart-base \
    --epochs_to_generate 10 \
    --max_source_len ${max_source_len} \
    --max_target_len ${max_target_len} \
    --max_context_len ${max_context_len} \
    --world_size ${worldsize} \
    --dev_num 0 \
    --already_preprocessed tapex \
    --global_rank $i &
done
wait
