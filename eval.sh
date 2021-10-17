#!/usr/bin/env bash

task=wtqqa

host=$(hostname)

if [[ "$host" == "GPU02" ]]; then
  prefix=$HOME
else
  prefix=""
fi

pred=$1

if [[ "$task" == "wtqqa" ]]; then
  prep_file=${prefix}/mnt/root/TaBERT/data/wikitablequestions/test/preprocessed_with_ans.jsonl
else
  exit
fi

python -m utils.eval \
  --prediction ${pred} \
  --prep_file ${prep_file}
