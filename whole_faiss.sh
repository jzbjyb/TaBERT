#!/usr/bin/env bash

source initialize.sh

path=$1

python -m utils.generate_grappa_data \
    --data whole_faiss \
    --path ${path}
