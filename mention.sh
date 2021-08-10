#!/usr/bin/env bash

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
  python -m utils.generate_grappa_data --data retpair \
    --path ${entry} ${target_file} ${source_file} \
    --output_dir ${out_file}.${eb} \
    --split context
done

# combine results
readarray -d '' entries < <(printf '%s\0' ${out_file}.* | sort -zV)  # sort
cat ${entries[@]} > ${out_file}
