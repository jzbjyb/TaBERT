#!/usr/bin/env bash
set -e

root_dir=$1
collection=$2
epoch=50

if [[ "$collection" == cate_efp_* ]]; then
  topics=(Q1457982 Q4047087 Q54070 Q1457673 Q8255 Q1458390 Q6337045 Q7214908 Q2944929 Q1457595)
  names=(Sports People Government Geography Music MassMedia Entertainment Events Culture History)
  prefix=wtqqa_strict_efp_
  if [[ "$collection" == "cate_efp_fewshot" ]]; then
    suffix=_128_ep${epoch}
  else
    suffix=_ep${epoch}
  fi
elif [[ "$collection" == topic_fp_* ]]; then
  topics=(sports culture people politics misc)
  names=(Sports Culture People Politics Misc)
  prefix=wtqqa_strict_fp_
  if [[ "$collection" == "topic_fp_fewshot" ]]; then
    suffix=_128_ep${epoch}
  else
    suffix=_ep${epoch}
  fi
fi

# header
echo -en 'train/test\t'
for j in "${!topics[@]}"; do
  echo -en ${names[j]} '\t'
done
echo ''

# content
for i in "${!topics[@]}"; do
  train_topic=${topics[i]}
  train_name=${names[i]}
  echo -en ${train_name} '\t'
  for j in "${!topics[@]}"; do
    test_topic=${topics[j]}
    test_name=${names[j]}
    file=${root_dir}_${prefix}${train_topic}${suffix}/ep$((epoch - 1)).tsv.test_${test_topic}
    result=$(./eval.sh ${file} 2> /dev/null | head -n 1)
    echo -en ${result} '\t'
  done
  echo ""
done
