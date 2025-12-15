#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.
# Kun: updated to install DiskANN to system directly, and update base image

FROM ubuntu:24.04

RUN apt update
RUN apt install -y software-properties-common git make cmake g++ libaio-dev \
    libgoogle-perftools-dev libunwind-dev clang-format libboost-dev \
    libboost-program-options-dev libmkl-full-dev libcpprest-dev

WORKDIR /app
RUN cd /app && git clone https://github.com/rmit-ir/DiskANN
WORKDIR /app/DiskANN
RUN cd /app/DiskANN && mkdir build
RUN cd /app/DiskANN/build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release .. && make -j && make install
