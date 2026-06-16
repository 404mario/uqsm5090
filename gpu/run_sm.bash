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
# The only CUDA toolkit deps (libcublas.so.13 + its transitive libcublasLt.so.13)
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
	if [ ! -f "$SCRIPT_DIR/lib/libcublas.so.13" ] || [ ! -f "$SCRIPT_DIR/lib/libcublasLt.so.13" ]; then
		echo "vendored CUDA libs missing under $SCRIPT_DIR/lib/ (need libcublas.so.13 + libcublasLt.so.13)"
		echo "Fetch with: $SCRIPT_DIR/setup_libs.sh"
		echo "===GPU Stress Test Failed==="
		return 1
	fi

	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "nvidia-smi not found"
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
