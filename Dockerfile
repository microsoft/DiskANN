FROM ubuntu:16.04
MAINTAINER Changxu Wang <wang_changxu@zju.edu.cn>

RUN apt-get update -y
RUN apt-get install -y g++ cmake libboost-dev libgoogle-perftools-dev

COPY . /opt/nsg

WORKDIR /opt/nsg

RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j $(nproc)
