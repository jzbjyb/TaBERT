#!/usr/bin/env bash
set -e

# with SQL and WTQ as placeholder
file_temp=$1  # /mnt/root/TaBERT/data/runs/wholetable_tapex_05m_wtqnl_denormalizedSQL_bart_qa_tapexbaseinit_wtqqa_strictWTQ_ep50/ep49.tsv.0
nums=(16 32 64 128 256 512 1024 "")

# header
echo -en 'wtq/sql\t'
for j in "${!nums[@]}"; do
  echo -en ${nums[j]} '\t'
done
echo ''

# content
for i in "${!nums[@]}"; do
  wtq_num=${nums[i]}
  echo -en ${wtq_num} '\t'
  if [[ ${wtq_num} != "" ]]; then
    wtq_num=_${wtq_num}
  fi
  for j in "${!nums[@]}"; do
    sql_num=${nums[j]}
    if [[ ${sql_num} != "" ]]; then
      sql_num=_num${sql_num}
    fi
    file=$(echo ${file_temp} | sed "s/SQL/${sql_num}/")
    file=$(echo ${file} | sed "s/WTQ/${wtq_num}/")
    result=$(./eval.sh ${file} 2> /dev/null | head -n 1)
    echo -en ${result} '\t'
  done
  echo ""
done
