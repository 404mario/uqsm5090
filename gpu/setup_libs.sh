#!/bin/bash
# Fetch the CUDA user-space libs that gpu_burn needs, WITHOUT docker or a CUDA
# toolkit. Places libcublas.so.<MAJOR> + libcublasLt.so.<MAJOR> into gpu/lib/.
#
# We vendor the **CUDA 12.8** cuBLAS. 12.8 is the first toolkit that supports
# Blackwell / RTX 5090 (sm_120) and only needs NVIDIA driver >= 570 -- yet it
# also runs on every newer driver. A CUDA 13 build, by contrast, needs driver
# >= 580 and silently kills every gpu_burn worker on a 570-era host. Building
# against 12.8 makes one artifact universal across the whole 5090 fleet.
# See run_sm.bash (preflight) and CLAUDE.md for the full rationale.
#
# These two libs are >100 MB combined (libcublasLt alone is ~500 MB), so they
# are NOT committed to git. The published OSS tarball already bundles them; this
# script is only needed after a fresh `git clone`.
#
# The CUDA *driver* lib (libcuda.so.1) is NOT fetched here -- it is provided by
# the installed NVIDIA driver and must never be bundled.
#
# Usage: ./setup_libs.sh
set -euo pipefail

LIBDIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)/lib"
mkdir -p "$LIBDIR"

# CUDA major the vendored libs belong to (soname suffix). Override CUDA_MAJOR to
# match a different wheel if you ever change CUBLAS_WHEEL_URL.
CUDA_MAJOR="${CUDA_MAJOR:-12}"
CUDA_VER="${CUDA_VER:-12.8}"

if [ -f "$LIBDIR/libcublas.so.$CUDA_MAJOR" ] && [ -f "$LIBDIR/libcublasLt.so.$CUDA_MAJOR" ]; then
	echo "[setup_libs] libs already present in $LIBDIR -- nothing to do."
	exit 0
fi

# CUDA 12.8 cuBLAS, shipped as a plain pip wheel (a ZIP) on PyPI's CDN.
# No pip/docker/toolkit required: download with curl, unpack with python's zipfile.
WHEEL_URL="${CUBLAS_WHEEL_URL:-https://files.pythonhosted.org/packages/8f/b0/343560086301700f0009f8155413771e775e6e2af5bb8d73e2b895c08159/nvidia_cublas_cu12-12.8.5.5-py3-none-manylinux_2_27_x86_64.whl}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[setup_libs] downloading CUDA $CUDA_VER cuBLAS wheel (~400 MB) ..."
curl -fL --retry 3 -C - -o "$TMP/cublas.whl" "$WHEEL_URL"

echo "[setup_libs] extracting libcublas.so.$CUDA_MAJOR + libcublasLt.so.$CUDA_MAJOR -> $LIBDIR"
# Locate the .so members by basename rather than hardcoding the in-zip path:
# cu12 wheels nest under nvidia/cublas/lib/, cu13 under nvidia/cu13/lib/, etc.
python3 - "$TMP/cublas.whl" "$LIBDIR" "$CUDA_MAJOR" <<'PY'
import sys, zipfile, shutil, os, re
whl, out, major = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(whl)
want = (f"libcublas.so.{major}", f"libcublasLt.so.{major}")
found = {}
for member in z.namelist():
    base = os.path.basename(member)
    if base in want:
        found[base] = member
missing = [w for w in want if w not in found]
if missing:
    sys.exit(f"[setup_libs] ERROR: {missing} not found in wheel; "
             f"check CUBLAS_WHEEL_URL / CUDA_MAJOR. zip has: "
             + ", ".join(sorted({os.path.basename(n) for n in z.namelist() if 'libcublas' in n})))
for base, member in found.items():
    dst = os.path.join(out, base)
    with z.open(member) as src, open(dst, "wb") as f:
        shutil.copyfileobj(src, f)
    print("  ", dst, os.path.getsize(dst), "bytes")
PY

# Record which CUDA the libs were built against; run_sm.bash reads this for the
# driver preflight, and it documents the artifact for anyone inspecting gpu/lib/.
echo "$CUDA_VER" > "$LIBDIR/.cuda_version"

echo "[setup_libs] done (CUDA $CUDA_VER). Contents of $LIBDIR:"
ls -lh "$LIBDIR"
