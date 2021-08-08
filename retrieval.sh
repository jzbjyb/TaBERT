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

echo 'download elasticsearch'
pushd ~/
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.14.0-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.14.0-linux-x86_64.tar.gz
cd elasticsearch-7.14.0
nohup bin/elasticsearch &
sleep 1m  # wait elasticsearch to start

popd
echo 'build index'
python -m table_bert.retrieval tapas-index

echo 'retrieval'
nthreads=$1
rank=$2
worldsize=$3
python -m table_bert.retrieval wikipedia-extend $nthreads $rank $worldsize
