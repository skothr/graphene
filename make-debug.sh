#!/usr/bin/env bash
mkdir -p build && cd build &&
    cmake -DCMAKE_BUILD_TYPE=Debug .. &&
    VERBOSE=1 make -j12 &&
    cp graphene ..
