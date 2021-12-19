#!/usr/bin/env bash

source env_initialize.sh

input_dir=data/totto_data/totto_train_data.official_preprocessed.jsonl
output_dir=/mnt/root/TaBERT/data/train_data/totto_data2text_official_bart
# --no_shuffle is needed for dev/test
max_source_len=1024
max_target_len=1024
max_context_len=128
mkdir -p ${output_dir}
worldsize=20

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name facebook/bart-base \
    --epochs_to_generate 1 \
    --max_source_len ${max_source_len} \
    --max_target_len ${max_target_len} \
    --max_context_len ${max_context_len} \
    --world_size ${worldsize} \
    --dev_num 0 \
    --already_preprocessed totto \
    --global_rank $i &
done
wait
