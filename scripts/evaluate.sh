#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHAMELION_DATASET_PATH:?Set the host dataset root or sequences directory}"
: "${CHAMELION_OUTPUT_PATH:?Choose a new host output directory}"
[[ $# -ge 1 ]] || { echo "Usage: bash scripts/evaluate.sh CHECKPOINT --sequence SEQUENCE [options]"; exit 2; }
checkpoint="$(realpath "$1")"
shift
dataset="$(realpath "$CHAMELION_DATASET_PATH")"
output="$(realpath -m "$CHAMELION_OUTPUT_PATH")"
[[ -f "$checkpoint" && -d "$dataset" ]] || exit 2
[[ ! -e "$output" && ! -L "$output" ]] || { echo "Output exists; choose a new directory"; exit 2; }
mkdir -p "$(dirname "$output")"
mkdir "$output"
echo "New results: $output/results"
docker run --rm --network none --user "$(id -u):$(id -g)" --gpus all --shm-size=2g \
  --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src \
  --volume "$repo_dir:/workspace:ro" --volume "$dataset:/input/data:ro" \
  --volume "$checkpoint:/input/model.ckpt:ro" --volume "$output:/output" \
  --workdir /workspace "${CHAMELION_IMAGE:-chamelion:train-cu128}" \
  python3 -m chamelion.evaluation.evaluate --checkpoint /input/model.ckpt \
  --data /input/data --output /output/results "$@"
