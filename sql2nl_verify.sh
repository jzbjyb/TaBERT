#!/usr/bin/env bash

source env_initialize.sh

num_gpu=$1
case=$2  # 16 128 1024
model=$3  # 128: wholetable_tapas_samepagehardmatchmax_tapex_sqlandnl_num128_large_wtqqa_strict_128_ep50 bart_large_wtqqa_strict_128_ep50
          # 16: tapex_large_wtqqa_strict_16_ep50
          # 1024: tapex_large_wtqqa_strict_1024_ep50
alias=$4

to_skip_file=/mnt/root/TaBERT/data/wikitablequestions/tapex/train.src.${case}
nl_prep_file=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num${case}.beamsearchnl
nl_tabert_dir=wholetable_wtq_wtqnl_denormalized_num${case}_beamsearch  # to be verified
out_file=/mnt/root/TaBERT/data/train_data/${nl_tabert_dir}/${model}/loss.txt

./run_model.sh ${num_gpu} large \
	${nl_tabert_dir} \
  ${model} \
  ${model}/pytorch_model_epoch49.bin \
  8 1 1 \
  --only_test \
  --mode computeloss-train \
  --output_file ${out_file}

filter_file=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num${case}.beamsearchnl.${alias}_filtered

python -m utils.prep --task select_by_loss \
  --inp ${out_file} ${nl_prep_file} ${to_skip_file} \
  --out ${filter_file} \
  --other ${num_gpu} \
