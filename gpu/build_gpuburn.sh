#!/bin/bash
# Build gpu-burn for RTX 5090 (sm_120) inside a CUDA 13 container.
# Run as: sudo ./build_gpuburn.sh
set -euo pipefail

OUT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
UID_GID="$(id -u mario):$(id -g mario)"

# NOTE: Docker is used ONLY to BUILD the binary (the host has no CUDA toolkit).
# The resulting gpu_burn runs on bare metal with no docker -- see run_sm.bash /
# setup_libs.sh. We bake an $ORIGIN/lib RPATH so the binary finds the vendored
# CUDA libs in gpu/lib/ relative to itself, regardless of where the dir is copied.
docker run --rm -v "$OUT:/out" nvidia/cuda:13.0.1-devel-ubuntu22.04 bash -c '
  set -e
  apt-get update -qq
  apt-get install -y -qq git ca-certificates >/dev/null
  cd /tmp
  git clone --depth 1 https://github.com/wilicc/gpu-burn
  cd gpu-burn
  make COMPUTE=120 LDFLAGS="-Wl,-rpath,\$ORIGIN/lib -lcuda -lcublas"
  # Upstream switched the comparison kernel from PTX to fatbin; copy whichever exists.
  cp gpu_burn /out/
  [ -f compare.fatbin ] && cp compare.fatbin /out/ || true
  [ -f compare.ptx ]    && cp compare.ptx    /out/ || true
'
chown "$UID_GID" "$OUT"/gpu_burn "$OUT"/compare.fatbin 2>/dev/null || true
chown "$UID_GID" "$OUT"/compare.ptx 2>/dev/null || true
echo "Built artifacts in $OUT:"
ls -l "$OUT"/gpu_burn "$OUT"/compare.fatbin "$OUT"/compare.ptx 2>/dev/null || true
