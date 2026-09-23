#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHAMELION_DATASET_PATH:?Set CHAMELION_DATASET_PATH to the host sequences directory}"

dataset_path="$(realpath "${CHAMELION_DATASET_PATH}")"
output_path="$(realpath -m "${CHAMELION_OUTPUT_PATH:-${repo_dir}/artifacts}")"
image_name="${CHAMELION_IMAGE:-chamelion:train-cu128}"
config_path="${CHAMELION_CONFIG_PATH:-config/cham.yaml}"
gpu_id="${CHAMELION_GPU_ID:-0}"
detach="${CHAMELION_DETACH:-0}"
container_name="${CHAMELION_CONTAINER_NAME:-chamelion-train}"

if [[ ! -d "${dataset_path}" ]]; then
  echo "Dataset directory does not exist: ${dataset_path}" >&2
  exit 2
fi

mkdir -p "${output_path}"

run_options=(run)
if [[ "${detach}" == "1" ]]; then
  run_options+=(--detach --name "${container_name}")
else
  run_options+=(--rm)
fi

docker "${run_options[@]}" \
  --gpus "device=${gpu_id}" \
  --shm-size=32g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env PYTHONPATH=/workspace/chamelion/src \
  --volume "${repo_dir}:/workspace/chamelion:ro" \
  --volume "${dataset_path}:/data/Const_pseudo_dataset/sequences:ro" \
  --volume "${output_path}:/output" \
  --workdir /workspace/chamelion \
  "${image_name}" \
  python3 -m chamelion.training.train --config "${config_path}" "$@"
