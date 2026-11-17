#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
"${NVCC:-nvcc}" -O3 -std=c++17 -arch=native -Xcompiler -fPIC \
  --shared --cudart shared -Iinclude \
  src/full_sort.cu src/hierarchical_topk.cu src/radix_topk.cu src/sampler.cu \
  src/shared_comparison_api.cu -o build/libshared_comparison_python.so
echo "Built build/libshared_comparison_python.so"
