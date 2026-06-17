#!/bin/bash
# Build the C++ reward extension
# Usage: bash grpo/reward/build_ext.sh
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
c++ -O3 -shared -fPIC -std=c++17 \
    -I${PYBIND_INC} -I${PYTHON_INC} \
    grpo/reward/fast_reward.cpp \
    -o grpo/reward/fast_reward.so
echo "Built successfully!"
