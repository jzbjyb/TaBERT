#!/usr/bin/env bash

set -e

inp_root=/mnt/root/TaBERT/data/wikitablequestions/tapex/topic
out_root=/mnt/root/TaBERT/data/train_data

for topic in culture people politics sports; do
  # full
  ./gen_wtq_tapex_strict.sh ${inp_root}/valid.src.${topic} ${out_root}/wtq_qa_tapex_strict_1024_${topic}_dev true &> /dev/null
  ./gen_wtq_tapex_strict.sh ${inp_root}/test.src.${topic} ${out_root}/wtq_qa_tapex_strict_1024_${topic}_test true &> /dev/null
  #./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic}.exclude ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude false &> /dev/null  # exclusive data
  ./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic} ${out_root}/wtq_qa_tapex_strict_1024_${topic} false &> /dev/null  # inclusive data
  #rm -rf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude/dev
  rm -rf ${out_root}/wtq_qa_tapex_strict_1024_${topic}/dev
  #mv ${out_root}/wtq_qa_tapex_strict_1024_${topic}_dev/train_noshuf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude/dev
  #mv ${out_root}/wtq_qa_tapex_strict_1024_${topic}_test/train_noshuf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude/test
  mv ${out_root}/wtq_qa_tapex_strict_1024_${topic}_dev/train_noshuf ${out_root}/wtq_qa_tapex_strict_1024_${topic}/dev
  mv ${out_root}/wtq_qa_tapex_strict_1024_${topic}_test/train_noshuf ${out_root}/wtq_qa_tapex_strict_1024_${topic}/test
  rm -rf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_dev
  rm -rf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_test
  # few-shot
  for num in 16 128 1024; do
    #./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic}.exclude.${num} ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude_num${num} false &> /dev/null  # exclusive data
    ./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic}.${num} ${out_root}/wtq_qa_tapex_strict_1024_${topic}_num${num} false &> /dev/null  # inclusive data
    #rm -rf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude_num${num}/dev
    rm -rf ${out_root}/wtq_qa_tapex_strict_1024_${topic}_num${num}/dev
    #cp -r ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude/dev ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude_num${num}/.
    #cp -r ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude/test ${out_root}/wtq_qa_tapex_strict_1024_${topic}_exclude_num${num}/.
    cp -r ${out_root}/wtq_qa_tapex_strict_1024_${topic}/dev ${out_root}/wtq_qa_tapex_strict_1024_${topic}_num${num}/.
    cp -r ${out_root}/wtq_qa_tapex_strict_1024_${topic}/test ${out_root}/wtq_qa_tapex_strict_1024_${topic}_num${num}/.
  done
done
