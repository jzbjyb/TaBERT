input_dir=data/train_data/vanilla_tabert
output_dir=data/runs/vanilla_tabert
mkdir -p ${output_dir}
batchsize=32
gradac=1
epochs=10

# (1) single node w/o deepspeed
# export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
# (2) single node w/ deepspeed
# export NGPU=8; deepspeed train.py
# (3) multi node w/ deepspeed
# export NGPU=8; deepspeed --hostfile=myhostfile train.py
export NGPU=8; deepspeed train.py \
    --task vanilla \
    --data-dir ${input_dir} \
    --output-dir ${output_dir} \
    --table-bert-extra-config '{}' \
    --train-batch-size ${batchsize} \
    --gradient-accumulation-steps ${gradac} \
    --learning-rate 2e-5 \
    --max-epoch ${epochs} \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128
