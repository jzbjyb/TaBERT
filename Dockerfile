FROM continuumio/anaconda3

WORKDIR '/app'

RUN export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0"
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install g++
COPY . .
RUN conda env create --file scripts/env.yml

SHELL ["conda", "run", "-n", "tabert", "/bin/bash", "-c"]
RUN pip install --editable .

CMD [ "sh", "--login", "-i" ]
