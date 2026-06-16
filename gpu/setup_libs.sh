#!/bin/bash
# Fetch the CUDA user-space libs that gpu_burn needs, WITHOUT docker or a CUDA
# toolkit. Places libcublas.so.13 + libcublasLt.so.13 into gpu/lib/.
#
# These two libs are >100 MB combined (libcublasLt.so.13 alone is ~514 MB), so
# they are NOT committed to git. The published OSS tarball already bundles them;
# this script is only needed after a fresh `git clone`.
#
# The CUDA *driver* lib (libcuda.so.1) is NOT fetched here -- it is provided by
# the installed NVIDIA driver and must never be bundled.
#
# Usage: ./setup_libs.sh
set -euo pipefail

LIBDIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)/lib"
mkdir -p "$LIBDIR"

if [ -f "$LIBDIR/libcublas.so.13" ] && [ -f "$LIBDIR/libcublasLt.so.13" ]; then
	echo "[setup_libs] libs already present in $LIBDIR -- nothing to do."
	exit 0
fi

# CUDA 13.0 cuBLAS, shipped as a plain pip wheel (a ZIP) on PyPI's CDN.
# No pip/docker/toolkit required: download with curl, unpack with python's zipfile.
WHEEL_URL="${CUBLAS_WHEEL_URL:-https://files.pythonhosted.org/packages/5a/99/210e113dde53955e97042bd76dc4ad927eca04c5b4645ec157cc59f4f3ae/nvidia_cublas-13.0.0.19-py3-none-manylinux_2_27_x86_64.whl}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[setup_libs] downloading cuBLAS wheel (~400 MB) ..."
curl -fL --retry 3 -C - -o "$TMP/cublas.whl" "$WHEEL_URL"

echo "[setup_libs] extracting libcublas.so.13 + libcublasLt.so.13 -> $LIBDIR"
python3 - "$TMP/cublas.whl" "$LIBDIR" <<'PY'
import sys, zipfile, shutil, os
whl, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
for member in ("nvidia/cu13/lib/libcublas.so.13", "nvidia/cu13/lib/libcublasLt.so.13"):
    dst = os.path.join(out, os.path.basename(member))
    with z.open(member) as src, open(dst, "wb") as f:
        shutil.copyfileobj(src, f)
    print("  ", dst, os.path.getsize(dst), "bytes")
PY

echo "[setup_libs] done. Contents of $LIBDIR:"
ls -lh "$LIBDIR"
