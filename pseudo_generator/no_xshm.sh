#!/usr/bin/env bash
# Run a command without X11 shared-memory image transfer. Works in an existing container.
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
shim_dir="$(mktemp -d /tmp/chamelion-xshm.XXXXXXXX)"
trap 'rm -f -- "${shim_dir}/disable_xshm.so"; rmdir -- "${shim_dir}"' EXIT
cc -shared -fPIC -Wall -Werror "${repo_dir}/pseudo_generator/disable_xshm.c" \
  -o "${shim_dir}/disable_xshm.so" -ldl
LD_PRELOAD="${shim_dir}/disable_xshm.so${LD_PRELOAD:+:${LD_PRELOAD}}" "$@"
