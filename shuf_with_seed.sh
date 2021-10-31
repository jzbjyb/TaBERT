#!/usr/bin/env bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

file=$1
shuf --random-source=<(get_seeded_random 2021) $file
