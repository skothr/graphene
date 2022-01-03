#!/bin/bash
env XMODIFIERS=@im=none emacs                       \
    ./src/*.cpp ./inc/*.hpp ./inc/*.h               \
    ./src/ui/*.cpp ./inc/ui/*.hpp ./inc/ui/*.h      \
    ./src/cuda/*.cu ./inc/cuda/*.cuh ./inc/cuda/*.h \
    &

