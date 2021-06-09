FROM continuumio/anaconda3

WORKDIR '/app'

# essential tools
RUN apt-get update
RUN apt-get -y install openssh-client vim tmux sudo apt-transport-https apt-utils curl \
    git wget lsb-release ca-certificates gnupg gcc g++

# TaBERT env
COPY . .
RUN export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0"
RUN conda env create --file scripts/env.yml
SHELL ["conda", "run", "-n", "tabert", "/bin/bash", "-c"]
RUN pip install --editable .
