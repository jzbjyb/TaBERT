FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# essential tools
RUN apt-get update
RUN apt-get -y install openssh-client vim tmux sudo apt-transport-https apt-utils curl \
    git wget lsb-release ca-certificates gnupg gcc g++

# Conda Environment
ENV MINICONDA_VERSION py37_4.9.2
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

WORKDIR '/app'

# TaBERT env
COPY scripts/env.yml /tmp/env.yml
RUN export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0"
RUN conda env create --file /tmp/env.yml
COPY . .
SHELL ["conda", "run", "-n", "tabert", "/bin/bash", "-c"]
RUN pip install --editable .

# deepspeed
RUN pip install deepspeed
