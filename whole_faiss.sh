#!/usr/bin/env bash

source initialize.sh

task=$1
input=$2
output=$3

python -m utils.generate_grappa_data \
    --data ${task} \
    --path ${input} \
    --output_dir ${output}
