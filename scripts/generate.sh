#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/generate.sh DATASET_ROOT SOURCE_SEQUENCE [--check]" >&2
  exit 2
fi
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="$(realpath "$1")"
sequence="$2"
check_only="${3:-}"
if [[ $# -gt 3 || ( -n "${check_only}" && "${check_only}" != --check ) ]]; then
  echo "Only the optional --check flag is supported; arbitrary output paths are not accepted." >&2
  exit 2
fi
if [[ ! "${sequence}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Sequence must be a single directory name." >&2
  exit 2
fi
source_dir="${dataset_root}/sequences/${sequence}"
if [[ ! -f "${source_dir}/poses.txt" || ( ! -d "${source_dir}/clouds" && ! -d "${source_dir}/velodyne" ) ]]; then
  echo "Expected sequences/${sequence}/{poses.txt,clouds/} inside DATASET_ROOT (the directory containing sequences/)." >&2
  exit 2
fi
if [[ ! -d "${source_dir}/clouds" ]]; then
  echo "Using legacy velodyne/ input without renaming it. New datasets should use clouds/." >&2
fi
: "${DISPLAY:?Run this from your graphical desktop terminal (DISPLAY is missing)}"
if [[ ! "${DISPLAY}" =~ ^:[0-9]+(\.[0-9]+)?$ ]]; then
  echo "A local X11/XWayland display is required, such as :0 or :1." >&2
  exit 2
fi
command -v xauth >/dev/null || { echo "Install xauth on the host first." >&2; exit 2; }
auth_source="${XAUTHORITY:-${HOME}/.Xauthority}"
if [[ ! -r "${auth_source}" ]]; then
  echo "X authority file is not readable: ${auth_source}" >&2
  exit 2
fi
auth_file="$(mktemp /tmp/chamelion-pseudo-xauth.XXXXXXXX)"
trap 'rm -f -- "${auth_file}"' EXIT
xauth -f "${auth_source}" nlist "${DISPLAY}" | sed 's/^..../ffff/' | xauth -f "${auth_file}" nmerge -
if [[ ! -s "${auth_file}" ]]; then
  echo "No X11 cookie found for ${DISPLAY}; set XAUTHORITY to your desktop session's authority file." >&2
  exit 2
fi

# The only writable host mount is a newly allocated, non-symlinked run directory.
run_dir="$(python3 - "${repo_dir}" <<'PY'
from pathlib import Path
import sys
import tempfile
root = Path(sys.argv[1]).resolve()
parent = root / "artifacts" / "pseudo_interactive"
for path in (root / "artifacts", parent):
    if path.is_symlink():
        raise SystemExit(f"Refusing symlinked output parent: {path}")
parent.mkdir(parents=True, exist_ok=True)
if parent.resolve() != parent:
    raise SystemExit("Output parent resolves outside the expected location")
print(tempfile.mkdtemp(prefix="run_", dir=parent))
PY
)"
echo "Read-only source: ${dataset_root}"
echo "NEW output directory: ${run_dir}"
run_options=(--rm --network none --user "$(id -u):$(id -g)" --shm-size=2g
  --env HOME=/tmp --env "DISPLAY=${DISPLAY}" --env XAUTHORITY=/tmp/pseudo.xauth
  --env LIBGL_ALWAYS_SOFTWARE=1 --env "PSEUDO_SOURCE_SEQUENCE=${sequence}"
  --volume "${repo_dir}:/workspace:ro" --volume "${dataset_root}:/input:ro"
  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro --volume "${auth_file}:/tmp/pseudo.xauth:ro"
  --volume "${run_dir}:/output" --workdir /workspace)
if [[ -t 0 && -t 1 ]]; then run_options+=(-it); fi
docker run "${run_options[@]}" "${CHAMELION_PSEUDO_IMAGE:-chamelion:pseudo-travel}" \
  bash pseudo_generator/no_xshm.sh bash pseudo_generator/session.sh "${check_only}"
