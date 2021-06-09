input_dir=data/train_data/vanilla_tabert
output_dir=data/runs/vanilla_tabert
mkdir -p ${output_dir}

export NGPU=4; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --task vanilla \
    --data-dir ${input_dir} \
    --output-dir ${output_dir} \
    --table-bert-extra-config '{}' \
    --train-batch-size 8 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --max-epoch 10 \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128
