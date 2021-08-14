#!/usr/bin/env bash
mkdir -p build && cd build &&
    cmake -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_BUILD_TYPE=Release .. &&
    make -j12 &&
    cp graphene ..
