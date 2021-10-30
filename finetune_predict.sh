#!/usr/bin/env bash

# --- arguments ---
only_predict=false  # use "true" if only perform prediction
task=$1  # "wtqqa"
model_size=$2  # "base", "large"
scale=$3  # "full" (8 gpus), "half" (4 gpus)

# hyperparameters
if [[ "$model_size" == "base" ]]; then
  if [[ "$scale" == "full" ]]; then
    ngpu=8
    batch_size=12
    grad_acc=1
  elif [[ "$scale" == "half" ]]; then
    ngpu=4
    batch_size=12
    grad_acc=2
  else
    exit 1
  fi
  base_model_name=facebook/bart-base
elif [[ "$model_size" == "large" ]]; then
  if [[ "$scale" == "full" ]]; then
    ngpu=8
    batch_size=6
    grad_acc=2
  elif [[ "$scale" == "half" ]]; then
    ngpu=4
    batch_size=6
    grad_acc=4
  else
    exit 1
  fi
  base_model_name=facebook/bart-large
else
  exit 1
fi

model_ckpt=$4  # use for initialization if it follows the pattern "*.bin"; otherwise use it as output directory
epoch=$5
args="${@:6}"  # additional args


# --- finetune & predict ---
IFS=':' read -ra tasks <<< "$task"
IFS=':' read -ra epochs <<< "$epoch"

for task in "${tasks[@]}"; do
  if [[ "$task" == "wtqqa" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_deprecate" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_firstansrow_add30
    mode=generate-test
  fi

  for epoch in "${epochs[@]}"; do
    if [[ "$only_predict" == "true" ]]; then
      output=${model_ckpt}
    else
      # finetune
      if [[ ${model_ckpt} == *.bin ]]; then   # a real checkpoint
        output="$(dirname "${model_ckpt}")"_${task}_ep${epoch}
        ./run_vanilla.sh ${ngpu} ${data} ${output} seq2seq ${batch_size} ${epoch} '"'${model_ckpt}'"' \
          --gradient-accumulation-steps ${grad_acc} --base-model-name ${base_model_name} \
          --mode ${mode} ${args}
      else
        output=${model_ckpt}_${task}_ep${epoch}
        ./run_vanilla.sh ${ngpu} ${data} ${output} seq2seq ${batch_size} ${epoch} null \
          --gradient-accumulation-steps ${grad_acc} --base-model-name ${base_model_name} \
          --mode ${mode} ${args}
      fi
    fi

    # evaluate every checkpoint
    remain=${ngpu}
    isfirst=true
    max_epoch=$(expr $epoch - 1)  # skip the last epoch because it's already evaluated
    for (( i=0; i<$max_epoch; ++i )); do
      iwz=$(printf "%02d" $i)  # add a preceding zero
      inter_ckpt=${output}/pytorch_model_epoch${iwz}.bin
      prediction_file=ep${iwz}.tsv
      CUDA_VISIBLE_DEVICES="$((remain - 1))" ./run_vanilla.sh \
        1 ${data} ${output} seq2seq ${batch_size} 1 '"'${inter_ckpt}'"' \
        --base-model-name ${base_model_name} \
        --only_test --mode ${mode} --output_file ${prediction_file} &
      if [[ "$isfirst" == "true" ]]; then
        # run the first exclusively to download necessary files
        wait
        isfirst=false
      fi
      remain="$((remain - 1))"
      if (( ${remain} == 0 )); then
        wait
        remain=${ngpu}
      fi
    done
    wait
  done
done