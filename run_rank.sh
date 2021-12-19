#!/usr/bin/env bash

source env_initialize.sh

input_dir=$1
model_path=$2
output_dir=$3
batchsize=$4
args="${@:5}"
epochs=5

#export NGPU=2; export NCCL_DEBUG=INFO; python -m torch.distributed.launch --nproc_per_node=$NGPU utils/rank.py \
python -m utils.rank \
    --sample_file ${input_dir}/samples.tsv \
    --db_file ${input_dir}/db_tabert.json \
    --model_path ${model_path} \
    --output_file ${output_dir} \
    --batch_size ${batchsize} \
    --learning-rate 2e-5 \
    --max-epoch ${epochs} \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --clip-norm 1.0 \
    ${args}
