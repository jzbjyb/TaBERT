#!/usr/bin/env bash

source initialize.sh

ret_file=$1
target_file=$2
source_file=$3
out_file=$4
splitcount=$5

# split the ret file
mkdir -p /tmp/retsplit/
split -l ${splitcount} --numeric-suffixes ${ret_file} /tmp/retsplit/
readarray -d '' entries < <(printf '%s\0' /tmp/retsplit/* | sort -zV)  # sort

echo 'split into' ${entries[@]}

for entry in "${entries[@]}"; do
  eb="$(basename $entry)"
  python -m utils.generate_grappa_data --data match_context_table \
    --path ${entry} ${target_file} ${source_file} \
    --output_dir ${out_file}.${eb} \
    --split context
done

# combine results
readarray -d '' entries < <(printf '%s\0' ${out_file}.* | sort -zV)  # sort
cat ${entries[@]} > ${out_file}
