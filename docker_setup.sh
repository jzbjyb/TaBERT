#!/usr/bin/env bash

# mount
wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse fuse

mkdir -p /mnt/root
mkdir -p /mnt/blobfusetmp

blobfuse /mnt/root \
  --tmp-path=/mnt/blobfusetmp \
  -o attr_timeout=240 \
  -o entry_timeout=240 \
  -o negative_timeout=120 \
  -o allow_root \
  --container-name=t-zhjiang \
  --log-level=LOG_DEBUG \
  --file-cache-timeout-in-seconds=120

# install tools
pip install -U amlt \
  --extra-index-url https://msrpypi.azurewebsites.net/stable/7e404de797f4e1eeca406c1739b00867 \
  --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65
