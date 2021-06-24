#!/usr/bin/env bash

input_dir=$1
model_path=$2
output_dir=$3
batchsize=32
epochs=2

# activate env if needed
if [[ "$PATH" == *"tabert"* ]]; then
  echo "tabert env activated"
else
  echo "tabert env not activated"
  conda_base=$(conda info --base)
  source ${conda_base}/etc/profile.d/conda.sh
  conda activate tabert
fi

python -m utils.rank \
    --sample_file ${input_dir}/samples.tsv \
    --db_file ${input_dir}/db_tabert.json \
    --model_path ${model_path} \
    --output_file ${output_dir} \
    --batch_size ${batchsize} \
    --learning-rate 2e-5 \
    --max-epoch ${epochs} \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --clip-norm 1.0 \
    --finetune
