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

task1=$1
task2=$2
nthreads=$3
rank=$4
worldsize=$5

popd
echo 'build index'
python -m table_bert.retrieval $task1
sleep 1m  # wait

echo 'retrieval'
python -m table_bert.retrieval $task2 $nthreads $rank $worldsize
