#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image_name="${CHAMELION_IMAGE:-chamelion:train-cu128}"
architectures="${CHAMELION_CUDA_ARCH_LIST:-}"
if [[ -z "${architectures}" ]] && command -v nvidia-smi >/dev/null; then
  architectures="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | paste -sd ';' -)" || architectures=""
fi
architectures="${architectures:-8.6;8.9;12.0}"
echo "Building ${image_name} for CUDA architectures: ${architectures}"

docker build \
  --build-arg "TORCH_CUDA_ARCH_LIST=${architectures}" \
  --build-arg "MAX_JOBS=${CHAMELION_BUILD_JOBS:-2}" \
  --tag "${image_name}" \
  --file "${repo_dir}/docker/training/Dockerfile.train" \
  "${repo_dir}"
