#!/usr/bin/env bash
mkdir -p build && cd build &&
    cmake -DCMAKE_BUILD_TYPE=DEBUG .. &&
    make -j12 && cp graphene ..
