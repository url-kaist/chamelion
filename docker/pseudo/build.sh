#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
expected="$(tr -d '\n' < "${repo_dir}/third_party/TRAVEL_REVISION")"
actual="$(git -C "${repo_dir}/third_party/TRAVEL" rev-parse HEAD)"
if [[ "${actual}" != "${expected}" ]] || [[ -n "$(git -C "${repo_dir}/third_party/TRAVEL" status --porcelain)" ]]; then
  echo "TRAVEL must be clean and checked out at ${expected}. Run git submodule update --init." >&2
  exit 1
fi
docker build -f "${repo_dir}/docker/pseudo/Dockerfile" \
  -t "${CHAMELION_PSEUDO_IMAGE:-chamelion:pseudo-travel}" "${repo_dir}"
