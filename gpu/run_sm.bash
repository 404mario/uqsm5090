#!/bin/bash
# GPU stress runner. Backend is chosen by $1 (dcgm|gpuburn). Duration in seconds is $2.
# Exits with one of these final lines (the caller greps the last line):
#   ===GPU Stress Test Success===
#   ===GPU Stress Test Failed===

set -u

TOOL="${1:-gpuburn}"
DURATION="${2:-600}"
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

detect_os() {
	. /etc/os-release 2>/dev/null || true
	echo "${ID:-unknown}"
}

# ----------------------------------------------------------------- dcgm ---
run_dcgm() {
	local os; os=$(detect_os)
	if [ "$os" = "ubuntu" ]; then
		if ls "$SCRIPT_DIR"/datacenter-gpu-manager-4-*_amd64.deb >/dev/null 2>&1; then
			sudo apt-get install -y "$SCRIPT_DIR"/datacenter-gpu-manager-4-core_4.5.2-1_amd64.deb \
			                         "$SCRIPT_DIR"/datacenter-gpu-manager-4-cuda13_4.5.2-1_amd64.deb \
				|| dpkg -i "$SCRIPT_DIR"/datacenter-gpu-manager-4-*_amd64.deb
		fi
	elif [ "$os" = "centos" ] || [ "$os" = "rhel" ]; then
		if ls "$SCRIPT_DIR"/datacenter-gpu-manager-*.rpm >/dev/null 2>&1; then
			sudo rpm -ivh "$SCRIPT_DIR"/datacenter-gpu-manager-*.rpm || true
		fi
	else
		echo "Unsupported OS for dcgm install: $os"
		echo "===GPU Stress Test Failed==="; return 1
	fi

	if ! command -v dcgmi >/dev/null 2>&1; then
		echo "dcgmi not found after install"
		echo "===GPU Stress Test Failed==="; return 1
	fi

	sudo pkill -x nv-hostengine 2>/dev/null || pkill -x nv-hostengine 2>/dev/null || true
	sudo nv-hostengine 2>/dev/null || nv-hostengine 2>/dev/null || true
	sleep 2

	if dcgmi diag -r 4; then
		echo "===GPU Stress Test Success==="
		return 0
	else
		echo "===GPU Stress Test Failed==="
		return 1
	fi
}

# -------------------------------------------------------------- gpu-burn ---
# Runs the host gpu_burn binary directly on the physical machine -- no Docker.
# The only CUDA toolkit deps (libcublas.so.12 + its transitive libcublasLt.so.12)
# are vendored in $SCRIPT_DIR/lib and resolved via LD_LIBRARY_PATH. The CUDA
# driver lib (libcuda.so.1) comes from the installed NVIDIA driver, as it always
# must -- it is the user-space half of the kernel module and cannot be bundled.
run_gpuburn_host() {
	pushd "$SCRIPT_DIR" >/dev/null
	LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
		"./gpu_burn" -d "$DURATION"
	local rc=$?
	popd >/dev/null
	return $rc
}

# Guard against the "(DIED!)" trap: a cuBLAS from CUDA major N cannot initialize
# on a driver whose newest supported CUDA major is < N (e.g. CUDA 13 cuBLAS on a
# driver that tops out at CUDA 12.8). cuInit still succeeds, but the first
# cublasCreate in each worker fails and gpu_burn just reports every GPU DIED with
# no hint why. Compare the vendored cuBLAS major against the driver's max CUDA
# and fail loudly with the fix. $1 = path to a vendored libcublas.so.<N>.
gpuburn_driver_preflight() {
	local cublas_lib="$1"
	local vmajor; vmajor="$(basename "$cublas_lib")"; vmajor="${vmajor##*.so.}"; vmajor="${vmajor%%.*}"
	local vstr="CUDA ${vmajor}.x"
	[ -f "$SCRIPT_DIR/lib/.cuda_version" ] && vstr="CUDA $(cat "$SCRIPT_DIR/lib/.cuda_version")"

	# nvidia-smi header prints "CUDA Version: X.Y" = the newest CUDA this driver supports.
	local dcuda; dcuda="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
	if [ -z "$dcuda" ]; then
		echo "[preflight] WARNING: could not read driver CUDA version from nvidia-smi; running anyway."
		return 0
	fi
	local dmajor="${dcuda%%.*}"
	if [ "$dmajor" -lt "$vmajor" ]; then
		echo "[preflight] FATAL driver/cuBLAS mismatch:"
		echo "    vendored cuBLAS  : $vstr  (gpu/lib/libcublas.so.$vmajor)"
		echo "    driver max CUDA  : $dcuda  ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1))"
		echo "    CUDA $vmajor cuBLAS needs a driver supporting CUDA ${vmajor}.x or newer; on this host"
		echo "    every gpu_burn worker would die at cublasCreate (the cryptic '(DIED!)')."
		echo "    Fix one of:"
		echo "      - upgrade the NVIDIA driver to one supporting CUDA ${vmajor}.x+, or"
		echo "      - rebuild gpu-burn for this driver:  CUDA_VER=${dcuda} CUDA_MAJOR=${dmajor} $SCRIPT_DIR/build_gpuburn.sh"
		echo "        then re-fetch matching libs:        CUDA_VER=${dcuda} CUDA_MAJOR=${dmajor} $SCRIPT_DIR/setup_libs.sh"
		return 1
	fi
	echo "[preflight] OK: driver supports CUDA $dcuda; vendored cuBLAS is $vstr."
	return 0
}

run_gpuburn() {
	local bin="$SCRIPT_DIR/gpu_burn"

	if [ ! -x "$bin" ]; then
		echo "gpu_burn binary missing under $SCRIPT_DIR/"
		echo "Build with: sudo $SCRIPT_DIR/build_gpuburn.sh"
		echo "===GPU Stress Test Failed==="
		return 1
	fi
	if [ ! -f "$SCRIPT_DIR/compare.fatbin" ] && [ ! -f "$SCRIPT_DIR/compare.ptx" ]; then
		echo "compare.fatbin (or compare.ptx) missing under $SCRIPT_DIR/"
		echo "Build with: sudo $SCRIPT_DIR/build_gpuburn.sh"
		echo "===GPU Stress Test Failed==="
		return 1
	fi
	# CUDA-major-agnostic: accept whatever libcublas.so.<N> is vendored (12, 13, ...)
	# instead of hardcoding one major. setup_libs.sh / build_gpuburn.sh currently
	# vendor CUDA 12.8 (libcublas.so.12) because 12.8 is the universal sm_120 floor.
	shopt -s nullglob
	local cublas_libs=("$SCRIPT_DIR"/lib/libcublas.so.[0-9]*)
	local cublasLt_libs=("$SCRIPT_DIR"/lib/libcublasLt.so.[0-9]*)
	shopt -u nullglob
	if [ ${#cublas_libs[@]} -eq 0 ] || [ ${#cublasLt_libs[@]} -eq 0 ]; then
		echo "vendored CUDA libs missing under $SCRIPT_DIR/lib/ (need libcublas.so.<N> + libcublasLt.so.<N>)"
		echo "Fetch with: $SCRIPT_DIR/setup_libs.sh"
		echo "===GPU Stress Test Failed==="
		return 1
	fi

	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "nvidia-smi not found"
		echo "===GPU Stress Test Failed==="; return 1
	fi

	# Preflight: a CUDA-N cuBLAS refuses to initialize on a driver whose max CUDA
	# is older than the toolkit it was built with -- and gpu_burn shows this only
	# as every worker "(DIED!)". Catch it here with an actionable message instead.
	if ! gpuburn_driver_preflight "${cublas_libs[0]}"; then
		echo "===GPU Stress Test Failed==="; return 1
	fi

	echo "[gpu-burn] running for ${DURATION}s on all visible GPUs (bare metal, no docker)..."
	local rc=0
	run_gpuburn_host; rc=$?

	if [ $rc -eq 0 ]; then
		echo "===GPU Stress Test Success==="
		return 0
	else
		echo "gpu_burn exit code: $rc"
		echo "===GPU Stress Test Failed==="
		return 1
	fi
}

case "$TOOL" in
	dcgm)    run_dcgm ;;
	gpuburn) run_gpuburn ;;
	*)
		echo "Unknown GPU tool: $TOOL (expected dcgm|gpuburn)"
		echo "===GPU Stress Test Failed==="
		exit 1
		;;
esac
