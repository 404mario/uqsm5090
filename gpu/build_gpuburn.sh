#!/bin/bash
# Build gpu-burn for RTX 5090 (sm_120) against CUDA 12.8 -- WITHOUT Docker, a
# CUDA toolkit, nvcc, or sudo. All it needs is g++, python3, and network access
# to PyPI. Run as:  ./build_gpuburn.sh
#
# Why 12.8 (not 13): 12.8 is the FIRST toolkit that supports Blackwell sm_120
# and only requires NVIDIA driver >= 570, while still running on every newer
# driver. A CUDA 13 build needs driver >= 580 and silently kills every gpu_burn
# worker (the "(DIED!)" failure) on a 570-era host. Building against 12.8 makes
# one bare-metal artifact universal across the whole 5090 fleet.
#
# How it works: gpu_burn is two pieces, BOTH rebuilt here against CUDA 12.8:
#   1. the C++ driver  (gpu_burn-drv.cpp) -- compiled with g++, links -lcublas.
#   2. the device kernel (compare.cu)     -- compiled with libnvrtc (the CUDA
#      runtime compiler) straight to an sm_120 SASS cubin, written to
#      gpu/compare.fatbin (cuModuleLoad accepts a raw cubin).
# All toolchain pieces come from NVIDIA's pip wheels (plain ZIPs on PyPI):
# cublas/runtime/nvcc/cccl headers, the cuBLAS .so, and libnvrtc.
#
# WHY we MUST recompile compare.cu (do NOT reuse a prebuilt one): a fatbin/cubin
# is toolchain-version-locked, NOT "CUDA-major-agnostic". The old committed file
# was built with CUDA 13's nvcc and embedded compute_120 *PTX* whose ISA version
# the 570/CUDA-12.8 driver's JIT rejects -> cuModuleLoad returns 222
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION -> every worker "(DIED!)". Compiling here
# with the 12.8 nvrtc to *sm_120 SASS* removes the PTX entirely, so there is
# nothing for an older driver to JIT. This is the second half of the 5090 fix.
set -euo pipefail

OUT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
LIBDIR="$OUT/lib"

CUDA_VER="${CUDA_VER:-12.8}"
CUDA_MAJOR="${CUDA_MAJOR:-12}"
COMPUTE="${COMPUTE:-120}"   # GPU arch for the compare kernel; 120 = Blackwell sm_120 (RTX 5090)
# Pinned wheel versions (all on PyPI as nvidia-*-cu12). Resolved to real URLs via
# the PyPI JSON API below, so file-hash churn does not break the script.
CUBLAS_VER="${CUBLAS_VER:-12.8.5.5}"
RUNTIME_VER="${RUNTIME_VER:-12.8.90}"   # cuda.h + driver headers
NVCC_VER="${NVCC_VER:-12.8.93}"         # crt/ headers (host_defines.h, ...)
CCCL_VER="${CCCL_VER:-12.8.90}"         # libcu++ (<nv/target>)
NVRTC_VER="${NVRTC_VER:-12.8.93}"       # libnvrtc -> compiles compare.cu to sm_120 SASS

for tool in g++ python3 curl; do
	command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' is required but not found." >&2; exit 1; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
INC="$WORK/include"; mkdir -p "$INC"

# Resolve the manylinux x86_64 wheel URL for a pinned nvidia-<pkg>-cu12 version.
wheel_url() {  # $1=package  $2=version
	python3 - "$1" "$2" <<'PY'
import sys, json, urllib.request
pkg, ver = sys.argv[1], sys.argv[2]
d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=60))
cands = [f["url"] for f in d["urls"]
         if f["filename"].endswith(".whl") and "x86_64" in f["filename"]]
if not cands:
    sys.exit(f"no x86_64 wheel for {pkg}=={ver}")
print(cands[0])
PY
}

fetch() {  # $1=package  $2=version  -> echoes local whl path
	local url path
	url="$(wheel_url "$1" "$2")"
	path="$WORK/$1-$2.whl"
	echo "[build] downloading $1 $2 ..." >&2
	curl -fL --retry 3 -o "$path" "$url"
	echo "$path"
}

# Extract every file under a wheel's .../include/ tree, preserving subdirs
# (crt/, cooperative_groups/, nv/, ...). Flattening breaks #include "crt/...".
extract_includes() {  # $1=whl  $2=destdir
	python3 - "$1" "$2" <<'PY'
import sys, zipfile, os, shutil
whl, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl); n = 0
for m in z.namelist():
    if "/include/" in m and not m.endswith("/"):
        rel = m.split("/include/", 1)[1]
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(m) as s, open(dst, "wb") as f:
            shutil.copyfileobj(s, f); n += 1
print(f"  +{n} headers from {os.path.basename(whl)}")
PY
}

echo "[build] assembling CUDA $CUDA_VER headers + cuBLAS libs from pip wheels (no docker/nvcc)"
CUBLAS_WHL="$(fetch nvidia-cublas-cu12      "$CUBLAS_VER")"
RUNTIME_WHL="$(fetch nvidia-cuda-runtime-cu12 "$RUNTIME_VER")"
NVCC_WHL="$(fetch nvidia-cuda-nvcc-cu12       "$NVCC_VER")"
CCCL_WHL="$(fetch nvidia-cuda-cccl-cu12       "$CCCL_VER")"
NVRTC_WHL="$(fetch nvidia-cuda-nvrtc-cu12      "$NVRTC_VER")"

extract_includes "$RUNTIME_WHL" "$INC"
extract_includes "$NVCC_WHL"    "$INC"
extract_includes "$CCCL_WHL"    "$INC"
extract_includes "$CUBLAS_WHL"  "$INC"

# cuBLAS .so live under .../lib/ in the wheel; pull them out by basename.
mkdir -p "$LIBDIR"
python3 - "$CUBLAS_WHL" "$LIBDIR" "$CUDA_MAJOR" <<'PY'
import sys, zipfile, os, shutil
whl, out, major = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(whl)
want = (f"libcublas.so.{major}", f"libcublasLt.so.{major}")
got = {}
for m in z.namelist():
    b = os.path.basename(m)
    if b in want:
        got[b] = m
missing = [w for w in want if w not in got]
if missing:
    sys.exit(f"missing {missing} in cuBLAS wheel")
for b, m in got.items():
    dst = os.path.join(out, b)
    with z.open(m) as s, open(dst, "wb") as f:
        shutil.copyfileobj(s, f)
    print("  lib:", b, os.path.getsize(dst), "bytes")
PY

# Unversioned link names so `-lcublas` resolves at build time (soname stays .12).
ln -sf "libcublas.so.$CUDA_MAJOR"   "$LIBDIR/libcublas.so"
ln -sf "libcublasLt.so.$CUDA_MAJOR" "$LIBDIR/libcublasLt.so"

# `-lcuda` needs a libcuda.so to resolve. The driver only guarantees
# libcuda.so.1; some hosts lack the dev symlink. Point at whatever libcuda.so.1
# the dynamic linker knows about, via a private symlink, so we never depend on
# the dev package being installed.
CUDA_DRV="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')"
[ -n "${CUDA_DRV:-}" ] || CUDA_DRV="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
if [ ! -e "$CUDA_DRV" ]; then
	echo "ERROR: libcuda.so.1 not found (is the NVIDIA driver installed?)." >&2
	exit 1
fi
STUB="$WORK/stub"; mkdir -p "$STUB"; ln -sf "$CUDA_DRV" "$STUB/libcuda.so"

echo "[build] cloning gpu-burn source"
git clone --depth 1 -q https://github.com/wilicc/gpu-burn "$WORK/gpu-burn"
cd "$WORK/gpu-burn"

echo "[build] g++ compile + link against libcublas.so.$CUDA_MAJOR"
g++ -I"$INC" -O3 -Wno-unused-result -c gpu_burn-drv.cpp -o gpu_burn-drv.o
# Bake $ORIGIN/lib RPATH so the binary finds gpu/lib/ relative to itself wherever
# the dir is copied. libcuda.so.1 is resolved from the system at runtime.
g++ -o gpu_burn gpu_burn-drv.o -O3 \
	-L"$LIBDIR" -lcuda -lcublas \
	-L"$STUB" \
	-Wl,-rpath,'$ORIGIN/lib'

cp gpu_burn "$OUT/gpu_burn"
echo "$CUDA_VER" > "$LIBDIR/.cuda_version"

# --- Regenerate the compare kernel with THIS toolchain (never reuse) ----------
# Compile compare.cu straight to an sm_120 SASS cubin via libnvrtc. SASS (not
# PTX) means an older driver has nothing to JIT, so it cannot raise
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION. gpu_burn's cuModuleLoad accepts a raw cubin,
# so we write it as compare.fatbin (the filename gpu_burn expects).
echo "[build] extracting libnvrtc $NVRTC_VER and recompiling compare.cu -> sm_120 SASS"
NVRTC_DIR="$WORK/nvrtc"; mkdir -p "$NVRTC_DIR"
python3 - "$NVRTC_WHL" "$NVRTC_DIR" <<'PY'
import sys, zipfile, os
whl, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
for m in z.namelist():
    # libnvrtc.so.12 plus its dlopen'd companion libnvrtc-builtins.so.12.8
    if "/lib/" in m and ".so" in os.path.basename(m):
        with z.open(m) as s, open(os.path.join(out, os.path.basename(m)), "wb") as f:
            f.write(s.read())
print("  nvrtc libs:", sorted(os.listdir(out)))
PY

ARCH="sm_$(echo "$COMPUTE" | tr -d '.')"
python3 - "$NVRTC_DIR/libnvrtc.so.12" "$WORK/gpu-burn/compare.cu" "$OUT/compare.fatbin" "$ARCH" <<'PY'
import ctypes, sys
nvrtc_path, src_path, out_path, arch = sys.argv[1:5]
n = ctypes.CDLL(nvrtc_path)
n.nvrtcGetErrorString.restype = ctypes.c_char_p
def chk(rc, where):
    if rc != 0:
        sys.exit(f"[build] nvrtc {where} failed: rc={rc} ({n.nvrtcGetErrorString(rc).decode()})")
src = open(src_path, "rb").read()
prog = ctypes.c_void_p()
chk(n.nvrtcCreateProgram(ctypes.byref(prog), src, b"compare.cu", 0, None, None), "create")
opts = [f"--gpu-architecture={arch}".encode()]      # SASS for this arch, no PTX
arr = (ctypes.c_char_p * len(opts))(*opts)
rc = n.nvrtcCompileProgram(prog, len(opts), arr)
lsz = ctypes.c_size_t(); n.nvrtcGetProgramLogSize(prog, ctypes.byref(lsz))
log = ctypes.create_string_buffer(lsz.value); n.nvrtcGetProgramLog(prog, log)
if log.value.strip(): print("  [nvrtc]", log.value.decode(errors="replace").strip())
chk(rc, "compile")
sz = ctypes.c_size_t(); chk(n.nvrtcGetCUBINSize(prog, ctypes.byref(sz)), "cubinsize")
buf = ctypes.create_string_buffer(sz.value); chk(n.nvrtcGetCUBIN(prog, buf), "cubin")
open(out_path, "wb").write(buf.raw[:sz.value])
print(f"  wrote {out_path} ({sz.value} bytes, {arch} SASS cubin)")
PY
# Sanity: must be an ELF cubin (SASS), not PTX text.
if ! head -c4 "$OUT/compare.fatbin" | grep -q $'\x7fELF'; then
	echo "ERROR: regenerated compare.fatbin is not an ELF cubin (PTX would re-introduce the bug)." >&2
	exit 1
fi
# ------------------------------------------------------------------------------

echo
echo "[build] done. Rebuilt gpu_burn (CUDA $CUDA_VER) AND regenerated compare.fatbin ($ARCH SASS)."
echo "[build] NEEDED libs:"
readelf -d "$OUT/gpu_burn" | grep -E 'NEEDED|RUNPATH' | sed 's/^/    /'
echo "[build] gpu/lib/:"
ls -lh "$LIBDIR" | sed 's/^/    /'
echo
echo "Verify:  LD_LIBRARY_PATH=$LIBDIR $OUT/gpu_burn -d 20"
echo "Then:    $OUT/run_sm.bash gpuburn 60   ->  ===GPU Stress Test Success==="

# ---------------------------------------------------------------------------
# Docker fallback (NOT normally needed -- this script already rebuilds both the
# binary and compare.fatbin toolkit-free via g++ + libnvrtc). Use it only if a
# host cannot reach PyPI but has Docker, or to cross-check against a full nvcc:
#
#   docker run --rm -v "$PWD:/out" nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c '
#     set -e; apt-get update -qq; apt-get install -y -qq git >/dev/null
#     cd /tmp && git clone --depth 1 https://github.com/wilicc/gpu-burn && cd gpu-burn
#     make COMPUTE=120 LDFLAGS="-Wl,-rpath,\$ORIGIN/lib -lcuda -lcublas"
#     cp gpu_burn compare.fatbin /out/'
# Whatever builds it, compare.fatbin MUST be sm_120 SASS (or 12.8-built) -- a
# CUDA-13 PTX-only fatbin will fail on a 570 driver with error 222.
# ---------------------------------------------------------------------------
