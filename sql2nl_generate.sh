#!/usr/bin/env bash

source env_initialize.sh

num_gpu=$1
case=$2  # 16 128 1024
model=wtq_sql2nl_denormalized_num${case}_bartlarge

gen_file=/mnt/root/TaBERT/data/train_data/wtq_sql2nl_denormalized_noshuf/wtq_sql2nl_denormalized_num${case}_beam/nl.txt

./run_model.sh ${num_gpu} large \
	wtq_sql2nl_denormalized_noshuf \
	${model} \
  ${model}/pytorch_model_epoch49.bin \
  8 1 1 \
  --only_test \
  --mode generate-train \
  --output_file ${gen_file} \
 	--num_return_sequences 50 \
 	--num_beams 50

prep_file=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized_withtable/train.src
full_prep_file=/mnt/root/TaBERT/data/wikitablequestions/tapex/train.src
nl_prep_file=/mnt/root/TaBERT/data/wikitablequestions/tapex/sql2nl_denormalized/train.src.num${case}.beamsearchnl

python -m utils.prep --task replace_context \
  --inp ${gen_file} ${prep_file} ${full_prep_file} \
  --out ${nl_prep_file} \
  --other ${num_gpu} True True
