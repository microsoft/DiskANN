#!/bin/bash

DEBIAN_FRONTEND=noninteractive apt install -y cmake \
  g++ \
  libaio-dev \
  libgoogle-perftools-dev \
  libunwind-dev \
  clang-format \
  libboost-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libmkl-full-dev