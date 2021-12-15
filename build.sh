#!/bin/bash
python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -- -j64

cd dist/ && pip uninstall * -y && pip install * && cd ..
