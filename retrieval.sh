#!/usr/bin/env bash

source env_initialize.sh

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
