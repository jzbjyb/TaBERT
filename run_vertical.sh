mkdir -p data/runs/vertical_tabert

export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
    --task vertical_attention \
    --data-dir data/train_data/vertical_tabert \
    --output-dir data/runs/vertical_tabert \
    --table-bert-extra-config '{"base_model_name": "bert-base-uncased", "num_vertical_attention_heads": 6, "num_vertical_layers": 3, "predict_cell_tokens": true}' \
    --train-batch-size 8 \
    --gradient-accumulation-steps 16 \
    --learning-rate 4e-5 \
    --max-epoch 1 \
    --adam-eps 1e-08 \
    --weight-decay 0.01 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128
