#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-training}" in
  training) exec bash "$repo_dir/docker/training/build.sh" ;;
  pseudo) exec bash "$repo_dir/docker/pseudo/build.sh" ;;
  viewer) exec docker build -f "$repo_dir/docker/viewer/Dockerfile" -t "${CHAMELION_VIEWER_IMAGE:-chamelion:viewer}" "$repo_dir" ;;
  *) echo "Usage: bash scripts/build.sh [training|pseudo|viewer]" >&2; exit 2 ;;
esac
