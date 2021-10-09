#!/usr/bin/env bash

ngpu=$1
task=$2  # from "wtqqa" "wikisqlqa" "turl_cf" "turl_sa"
model_dir=$3
ckpt=$4

epoch=1
batch_size=24
isfirst=true

remain=${ngpu}
IFS=':' read -ra tasks <<< "$task"
IFS=':' read -ra ckpts <<< "$ckpt"
for task in "${tasks[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        model_filename=pytorch_model_epoch${ckpt}.bin
        model_ckpt=${model_dir}/${model_filename}

        if [[ "$task" == "wtqqa" ]]; then
            data=/mnt/root/TaBERT/data/train_data/wtq_qa_firstansrow_add30
            mode=generate-test
        elif [[ "$task" == "wtqqa_tapex" ]]; then
            data=/mnt/root/TaBERT/data/train_data/wtq_qa_allrow
            mode=generate-test
        elif [[ "$task" == "wtqqa_tapex_large" ]]; then
            data=/mnt/root/TaBERT/data/train_data/wtq_qa_allrow_bart_large
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

        prediction_file=${model_filename%.*}_${mode}.tsv
        CUDA_VISIBLE_DEVICES="$((remain - 1))" ./run_vanilla.sh \
            1 ${data} ${model_dir} seq2seq ${batch_size} ${epoch} '"'${model_ckpt}'"' \
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
done
wait
