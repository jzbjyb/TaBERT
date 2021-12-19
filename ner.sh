#!/usr/bin/env bash

source env_initialize.sh

input=$1
output=$2

python -m spacy download en_core_web_sm
python -m utils.generate_grappa_data \
    --data ner \
    --path ${input} \
    --output_dir ${output}
