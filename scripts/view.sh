#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "Usage: bash scripts/view.sh evaluation RESULTS"
  echo "   or: bash scripts/view.sh sequence CHECKPOINT DATASET SEQUENCE custom|lista"
  echo "   or: bash scripts/view.sh inference CHECKPOINT MAP_PCD SCANS_DIR POSES_TXT global|local"
  exit 2
}
[[ $# -ge 1 ]] || usage
mode="$1"
options=()
software_gl=1
if [[ "$mode" == evaluation && $# == 2 ]]; then
  results="$(realpath "$2")"
  [[ -e "$results" ]] || usage
  options+=(--volume "$results:/input/results:ro")
  command=(python3 -m chamelion.visualization.results /input/results)
elif [[ "$mode" == sequence && $# == 5 ]]; then
  software_gl="${CHAMELION_SOFTWARE_GL:-0}"
  checkpoint="$(realpath "$2")"
  dataset="$(realpath "$3")"
  [[ -f "$checkpoint" && -d "$dataset" ]] || usage
  [[ "$5" == custom || "$5" == lista ]] || usage
  options+=(--gpus all --volume "$checkpoint:/input/model.ckpt:ro"
    --volume "$dataset:/input/data:ro"
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display)
  if [[ "$software_gl" == 0 ]]; then
    options+=(--env __GLX_VENDOR_LIBRARY_NAME=nvidia)
  fi
  command=(python3 -m chamelion.visualization.sequence --checkpoint /input/model.ckpt
    --data /input/data --sequence "$4" --profile "$5" --config /workspace/config/cham.yaml --output /output/viewer)
elif [[ "$mode" == inference && $# == 6 ]]; then
  checkpoint="$(realpath "$2")"
  prior="$(realpath "$3")"
  scans="$(realpath "$4")"
  poses="$(realpath "$5")"
  [[ -f "$checkpoint" && -f "$prior" && -d "$scans" && -f "$poses" ]] || usage
  [[ "$6" == global || "$6" == local ]] || usage
  options+=(--gpus all --volume "$checkpoint:/input/model.ckpt:ro"
    --volume "$prior:/input/prior.pcd:ro" --volume "$scans:/input/scans:ro"
    --volume "$poses:/input/poses.txt:ro")
  command=(python3 -m chamelion.visualization.interactive --checkpoint /input/model.ckpt
    --map /input/prior.pcd --scans /input/scans --poses /input/poses.txt
    --scan-coordinates "$6" --config /workspace/config/cham.yaml --output /output/predictions)
else
  usage
fi
: "${DISPLAY:?Run from a graphical desktop terminal}"
[[ "$DISPLAY" =~ ^:[0-9]+(\.[0-9]+)?$ ]] || { echo "A local X11/XWayland display is required"; exit 2; }
command -v xauth >/dev/null || { echo "Install xauth on the host first"; exit 2; }
auth_source="${XAUTHORITY:-${HOME}/.Xauthority}"
auth_file="$(mktemp /tmp/chamelion-viewer-xauth.XXXXXXXX)"
trap 'rm -f -- "$auth_file"' EXIT
xauth -f "$auth_source" nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$auth_file" nmerge -
[[ -s "$auth_file" ]] || { echo "No X11 cookie found for $DISPLAY"; exit 2; }
# Allocate only a fresh output directory. All inputs and repository files are read-only.
if [[ "$mode" == inference || "$mode" == sequence ]]; then
  [[ ! -L "$repo_dir/artifacts" && ! -L "$repo_dir/artifacts/viewer_runs" ]] || exit 2
  mkdir -p "$repo_dir/artifacts/viewer_runs"
  run_dir="$(mktemp -d "$repo_dir/artifacts/viewer_runs/run_XXXXXXXX")"
  options+=(--volume "$run_dir:/output")
  echo "New viewer session output: $run_dir"
fi
docker run --rm --init --network none --user "$(id -u):$(id -g)" --shm-size=2g \
  --env HOME=/tmp --env "DISPLAY=$DISPLAY" --env XAUTHORITY=/tmp/viewer.xauth \
  --env "LIBGL_ALWAYS_SOFTWARE=$software_gl" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/workspace/src \
  --volume "$repo_dir:/workspace:ro" --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --volume "$auth_file:/tmp/viewer.xauth:ro" --workdir /tmp \
  "${options[@]}" "${CHAMELION_VIEWER_IMAGE:-chamelion:viewer}" "${command[@]}"
