FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install \
    build-essential yasm nasm cmake \
    python3 python3-pip python3-dev python3-setuptools python3-opencv &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -sf /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN apt update
RUN apt install -y git

# Upgrade pip for cv package instalation
RUN pip3 install --upgrade pip==21.0.1

ENV LANG C.UTF-8

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install https://github.com/kpu/kenlm/archive/master.zip
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN pip3 install ./ctcdecode

WORKDIR /home/jovyan