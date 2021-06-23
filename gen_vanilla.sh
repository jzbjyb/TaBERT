output_dir=data/train_data/vanilla_tabert
input_dir=data/preprocessed_data/tables.jsonl
additional_row_count=0
mkdir -p ${output_dir}
worldsize=40

for (( i=0; i<${worldsize}; ++i)); do
  echo $i ${worldsize}
  python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${input_dir} \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 15 \
    --max_context_len 128 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column|type|value' \
    --column_delimiter "[SEP]" \
    --world_size ${worldsize} \
    --additional_row_count ${additional_row_count} \
    --global_rank $i >> ${output_dir}.out &
done
wait
