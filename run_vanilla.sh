#!/usr/bin/env bash

input_dir=$1  # data/train_data/vanilla_tabert
output_dir=$2  # data/runs/vanilla_tabert
mkdir -p ${output_dir}
loss=$3
batchsize=$4
epochs=$5
name="$(basename -- $output_dir)"
echo '==========' ${name} '=========='
gradac=1

# activate env if needed
if [[ "$PATH" == *"tabert"* ]]; then
  echo "tabert env activated"
else
  echo "tabert env not activated"
  conda_base=$(conda info --base)
  source ${conda_base}/etc/profile.d/conda.sh
  conda activate tabert
fi

# wandb
export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

# (1) single node w/o deepspeed
# export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
# (2) single node w/ deepspeed
# export NGPU=8; deepspeed train.py
# (3) single node w/ deepspeed and limited GPUs
# export NGPU=1; deepspeed --num_gpus 1 train.py
# (4) multi node w/ deepspeed
# export NGPU=8; deepspeed train.py
#export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
python train.py \
    --task vanilla \
    --data-dir ${input_dir} \
    --output-dir ${output_dir} \
    --table-bert-extra-config '{"objective_function": "'${loss}'"}' \
    --train-batch-size ${batchsize} \
    --gradient-accumulation-steps ${gradac} \
    --learning-rate 2e-5 \
    --max-epoch ${epochs} \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128 \
    --name ${name}
