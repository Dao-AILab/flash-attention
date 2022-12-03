#!/bin/bash

hipcc \
fmha_api.cpp \
-I/var/lib/jenkins/libtorch/include \
-I/var/lib/jenkins/libtorch/include/torch/csrc/api/include \
-I/usr/include/python3.8 \
-I${PWD}/src \
-I${PWD}/composable_kernel/include \
-I${PWD}/composable_kernel/library/include \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-std=c++17 \
-L/var/lib/jenkins/libtorch/lib \
-Wl,-R/var/lib/jenkins/libtorch/lib \
-Wl,-rpath-link=/usr/lib/x86_64-linux-gnu/ \
-Wl,--no-as-needed \
-ltorch -ltorch_cpu -lc10 -o fmha_api \
${PWD}/src/*.cpp \
${PWD}/composable_kernel/library/src/utility/*.cpp \
2>&1 | tee log.txt
