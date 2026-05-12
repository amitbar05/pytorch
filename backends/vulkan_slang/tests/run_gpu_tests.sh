#!/bin/bash
# Run tests on RDNA1 GPU hardware (RX 5600 XT at /dev/dri/renderD128).
#
# Usage:
#   ./run_gpu_tests.sh                    # run all GPU-tagged tests
#   ./run_gpu_tests.sh -k "conv_relu"     # filter by keyword
#   ./run_gpu_tests.sh -x                 # stop on first failure
#
# The script sets VK_ICD_FILENAMES to the Radeon ICD and passes --gpu
# to pytest so that the gpu_device fixture and collection-time marker
# filtering activate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json

cd "${REPO_ROOT}"

python -m pytest backends/vulkan_slang/tests/test_inductor_regression.py \
    -v --timeout=120 \
    --gpu \
    -m "gpu or both" \
    "$@"
