output_dir=/mnt/root/TaBERT/data/train_data/vanilla_tabert_3merge_seq2seq_maskseparate
input_dir=/mnt/root/TaBERT/data/grappa/totto_tablefact_wikisql_preprocessed.jsonl
additional_row_count=0
mkdir -p ${output_dir}
worldsize=10

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name facebook/bart-base \
    --do_lower_case \
    --epochs_to_generate 20 \
    --max_context_len 128 \
    --max_column_len 15 \
    --max_cell_len 15 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column | type | value' \
    --column_delimiter "[SEP]" \
    --world_size ${worldsize} \
    --additional_row_count ${additional_row_count} \
    --use_sampled_value \
    --mask_value \
    --mask_value_column_separate \
    --skip_column_name_longer_than 0 \
    --seq2seq_format mlm_single-c2v_single-v2c \
    --global_rank $i &
done
wait
