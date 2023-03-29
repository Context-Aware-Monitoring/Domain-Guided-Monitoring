FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
ARG CONDA_SH=Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
ARG CONDA_DIR=.tmp

RUN mkdir /Domain-Guided-Monitoring
RUN mkdir /Domain-Guided-Monitoring/artifacts
RUN mkdir /Domain-Guided-Monitoring/mlruns

COPY data /Domain-Guided-Monitoring/data
COPY notebooks /Domain-Guided-Monitoring/notebooks
COPY src /Domain-Guided-Monitoring/src
COPY tests /Domain-Guided-Monitoring/tests
COPY environment.yml /Domain-Guided-Monitoring
COPY r_install.sh /Domain-Guided-Monitoring
COPY main_refinement.py /Domain-Guided-Monitoring
COPY main.py /Domain-Guided-Monitoring
COPY Makefile /Domain-Guided-Monitoring

# Fixes Nvidia GPG key error
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Causality algorithms need GSL >=2.5
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
    apt install software-properties-common -y && \ 
    apt-get update && \
    add-apt-repository -y ppa:dns/gnu && \ 
    apt-get update && \
    apt install libgsl-dev -y

# DomainML and its dependencies installation
RUN cd /Domain-Guided-Monitoring && \
    mkdir .tmp && \
    apt-get install wget -y && \
    apt-get install nano -y && \
    wget -nc -P .tmp/ $CONDA_URL && \
    bash ./$CONDA_DIR/$CONDA_SH -b -u -p ./$CONDA_DIR/miniconda3/ && \
    .tmp/miniconda3/bin/conda config --set channel_priority flexible && \
    .tmp/miniconda3/bin/conda env update --file environment.yml --prune && \
    .tmp/miniconda3/bin/conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch && \
    bash r_install.sh