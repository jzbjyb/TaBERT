#!/usr/bin/env bash

inp=$1

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

# 16 32 64 128 256 512 1024
for num in 16 32 64 128 256 512 1024; do
  echo ${num}
  shuf --random-source=<(get_seeded_random ${num}) ${inp} | head -n ${num} > ${inp}.${num}
done
