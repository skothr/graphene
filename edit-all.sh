#!/bin/bash
env XMODIFIERS=@im=none emacs \
    ./src/* ./inc/*           \
    ./src/ui/* ./inc/ui/*     \
    ./src/cuda/* ./inc/cuda/* \
    &

