#!/usr/bin/env bash

set -e

#topics=(misc culture people politics sports)
#nums=(16 128 1024)

topics=(Q1457982 Q4047087 Q54070 Q1457673 Q8255 Q1458390 Q6337045 Q7214908 Q2944929 Q1457595 Q7386634 Q1457756 Q4103183 Q5613113 Q5850187 Q8413436 Q4049293 Q4103249 Q9715089 Q6528585 Q6353120 Q6576895 Q5645580 Q1458484 Q7486603)
nums=(16 128)

# our heuristc
#inp_root=/mnt/root/TaBERT/data/wikitablequestions/tapex/topic
#out_root=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_

# follow paper
#inp_root=/mnt/root/TaBERT/data/wikitablequestions/tapex/topic_followpaper
#out_root=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_

inp_root=/mnt/root/TaBERT/data/wikitablequestions/tapex/category_exclusive_followpaper
out_root=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

for topic in "${topics[@]}"; do
  echo ${topic}

  # full
  ./gen_wtq_tapex_strict.sh ${inp_root}/valid.src.${topic} ${out_root}${topic}_dev true &> /dev/null
  ./gen_wtq_tapex_strict.sh ${inp_root}/test.src.${topic} ${out_root}${topic}_test true &> /dev/null
  #./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic}.exclude ${out_root}${topic}_exclude false &> /dev/null  # exclusive data
  ./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic} ${out_root}${topic} false &> /dev/null  # inclusive data
  #rm -rf ${out_root}${topic}_exclude/dev
  rm -rf ${out_root}${topic}/dev
  #cp -r ${out_root}${topic}_dev/train_noshuf ${out_root}${topic}_exclude/dev
  #cp -r ${out_root}${topic}_test/train_noshuf ${out_root}${topic}_exclude/test
  cp -r ${out_root}${topic}_dev/train_noshuf ${out_root}${topic}/dev
  cp -r ${out_root}${topic}_test/train_noshuf ${out_root}${topic}/test
  rm -rf ${out_root}${topic}_dev
  rm -rf ${out_root}${topic}_test

  numsj=$(join_by : ${nums[@]})
  ./subsample.sh ${inp_root}/train.src.${topic} ${numsj}

  # few-shot
  for num in "${nums[@]}"; do
    #./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic}.exclude.${num} ${out_root}${topic}_exclude_num${num} false &> /dev/null  # exclusive data
    ./gen_wtq_tapex_strict.sh ${inp_root}/train.src.${topic}.${num} ${out_root}${topic}_num${num} false &> /dev/null  # inclusive data
    #rm -rf ${out_root}${topic}_exclude_num${num}/dev
    rm -rf ${out_root}${topic}_num${num}/dev
    #cp -r ${out_root}${topic}_exclude/dev ${out_root}${topic}_exclude_num${num}/.
    #cp -r ${out_root}${topic}_exclude/test ${out_root}${topic}_exclude_num${num}/.
    cp -r ${out_root}${topic}/dev ${out_root}${topic}_num${num}/.
    cp -r ${out_root}${topic}/test ${out_root}${topic}_num${num}/.
  done
done
