#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

FROM ubuntu:jammy

# Can be provided at build to point to a specific commit-ish, by default builds from HEAD
ARG GIT_COMMIT_ISH=HEAD

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y git time

COPY dev/install-dev-deps-ubuntu.bash /app/fallback/install-dev-deps-ubuntu.bash
WORKDIR /app
RUN git clone https://github.com/microsoft/DiskANN.git
WORKDIR /app/DiskANN
RUN git checkout $GIT_COMMIT_ISH

# we would prefer to use the deps requested at the same commit. if the script doesn't exist we'll use the current one.
RUN bash scripts/dev/install-dev-deps-ubuntu.bash || bash /app/fallback/install-dev-deps-ubuntu.bash

RUN mkdir build
RUN cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release -DUNIT_TEST=True
RUN cmake --build build -- -j

RUN mkdir /app/logs
COPY perf/perf_test.sh /app/DiskANN/perf_test.sh

ENTRYPOINT bash perf_test.sh
