#!/usr/bin/env bash

source initialize.sh

output=$1
input="${@:2}"

python -m utils.generate_grappa_data \
    --data ret_filter_by_faiss \
    --path ${input} \
    --output_dir ${output}
