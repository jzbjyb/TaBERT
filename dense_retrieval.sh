#!/usr/bin/env bash

source env_initialize.sh
pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
pip install absl-py==0.15.0
pip install datasets==1.15.1
pip install scikit-learn==0.24.2
python -m spacy download en_core_web_sm

model=$1
from=$2
to=$3
out=$4

mkdir -p $(dirname $out)

python -m utils.generate_sling_data \
  --task gen_sentence_emb \
  --inp /mnt/root/tapas/data/pretrain/train/preprocessed_mention.jsonl.data0 /mnt/root/sling/local/data/e/wiki \
  --other ${from} ${to} ${model} \
  --out ${out} \
  --threads 8
