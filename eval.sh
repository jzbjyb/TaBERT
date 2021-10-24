#!/usr/bin/env bash
set -e

task=wtqqa

host=$(hostname)
if [[ "$host" == "GPU02" ]]; then
  prefix=$HOME
else
  prefix=""
fi

pred=$1

if [[ "$task" == "wtqqa" ]]; then
  gold=${prefix}/mnt/root/TaBERT/data/wikitablequestions/test/preprocessed_with_ans.jsonl
else
  exit
fi

if [[ -d $pred ]]; then
  for i in ${pred}/*.tsv; do
    result=$(python -W ignore -m utils.eval --prediction ${i} --gold ${gold} --multi_ans_sep ", " 2> /dev/null | head -n 1)
    echo $(basename $i) ${result}
  done
elif [[ -f $pred ]]; then
  python -m utils.eval \
    --prediction ${pred} \
    --gold ${gold} \
    --multi_ans_sep ", "
else
  exit
fi
