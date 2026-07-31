FROM tensorflow/tensorflow:1.10.1-gpu-py3@sha256:4252dd3dd509e608c8722157aad1a5fedfcd3b76ad422dec8340b14dea6cbb7c
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN  apt-get install python-rdkit librdkit1 rdkit-data
RUN pip install scikit-learn
COPY . ./

RUN pip install .