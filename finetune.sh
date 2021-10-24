#!/usr/bin/env bash

ngpu=$1
task=$2  # from "wtqqa" "wikisqlqa" "turl_cf" "turl_sa"
model_ckpt=$3
batch_size=$4
args="${@:5}"  # additional args
epoch=10
prediction_file=ep9.tsv

IFS=':' read -ra tasks <<< "$task"
for task in "${tasks[@]}"; do
    output="$(dirname "${model_ckpt}")"_${task}

    if [[ "$task" == "wtqqa" ]]; then  # TODO: deprecated
        data=/mnt/root/TaBERT/data/train_data/wtq_qa_firstansrow_add30
        mode=generate-test
    elif [[ "$task" == "wtqqa_tapex" ]]; then
        data=/mnt/root/TaBERT/data/train_data/wtq_qa_allrow
        mode=generate-test
    elif [[ "$task" == "wikisqlqa" ]]; then
        data=/mnt/root/TaBERT/data/train_data/wikisql_qa_firstansrow_add30
        mode=generate-test
    elif [[ "$task" == "turl_cf" ]]; then
        data=/mnt/root/TaBERT/data/train_data/turl_cf_bart_mlm
        mode=evaluate-test
    elif [[ "$task" == "turl_sa" ]]; then
        data=/mnt/root/TaBERT/data/train_data/turl_sa_bart_mlm
        mode=evaluate-test
    else
        exit
    fi

    ./run_vanilla.sh ${ngpu} ${data} ${output} seq2seq ${batch_size} ${epoch} '"'${model_ckpt}'"' --mode ${mode} --output_file ${prediction_file} ${args}
done
