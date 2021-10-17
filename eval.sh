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
  gold=${prefix}/mnt/root/TaBERT/data/wikitablequestions/tapex/test.tgt
else
  exit
fi

python -m utils.eval \
  --prediction ${pred} \
  --gold ${gold} \
  --multi_ans_sep ", "
