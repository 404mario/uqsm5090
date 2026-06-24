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
# Runs gpu_burn and captures a filtered copy of its output to $1 for verdict
# parsing. gpu_burn floods the terminal with \r-overwritten progress lines (and
# a DIED worker repeats its error every cycle -> gigabytes of spam), so we
# convert \r to \n and throttle the progress lines, while ALWAYS keeping the
# summary, init errors, and any line containing "(DIED!)". Returns gpu_burn's
# own exit code via PIPESTATUS.
run_gpuburn_host() {
	local logfile="$1" rc
	pushd "$SCRIPT_DIR" >/dev/null
	LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
		"./gpu_burn" -d "$DURATION" 2>&1 \
		| tr '\r' '\n' \
		| awk '
			/proc.d:/ {                      # progress line: throttle the spam
				prog++
				if (prog % 200 == 0 || /\(DIED!\)/) { print; fflush() }
				next
			}
			{ print; fflush() }              # everything else passes through
		' \
		| tee "$logfile"
	rc=${PIPESTATUS[0]}
	popd >/dev/null
	return "$rc"
}

# When a worker dies, the usual cause on a shared host is too little free VRAM
# (another process already holds it). Surface per-GPU memory + the holders.
gpuburn_report_busy_gpus() {
	command -v nvidia-smi >/dev/null 2>&1 || return 0
	echo "[gpu-burn] gpu_burn allocates a large block of VRAM per GPU; deaths are"
	echo "           usually GPUs already in use. Current memory (MiB):"
	nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
	           --format=csv,noheader,nounits 2>/dev/null \
	  | awk -F', *' '{ printf "             GPU %s: %s/%s used, %s free%s\n", \
	                   $1,$2,$3,$4, ($4+0 < 2048 ? "   <-- too low for gpu_burn" : "") }'
	echo "[gpu-burn] processes holding GPU memory:"
	nvidia-smi --query-compute-apps=pid,process_name,used_memory \
	           --format=csv,noheader 2>/dev/null | sed 's/^/             /' || true
	echo "[gpu-burn] free the busy GPUs, or restrict to idle ones with"
	echo "           CUDA_VISIBLE_DEVICES=<ids>, then re-run."
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
	local rc=0 logfile
	logfile="$(mktemp)"
	run_gpuburn_host "$logfile"; rc=$?

	# IMPORTANT: do NOT trust gpu_burn's exit code or its "GPU N: OK" summary to
	# decide pass/fail. A worker that DIED has 0 compute errors, so gpu_burn marks
	# it OK and (unless ALL workers die) exits 0 -- a silent false pass. The only
	# reliable death signals are in the output itself, so we parse them here.
	local fail=0 reason="" died=0
	if grep -qaE '\(DIED!\)' "$logfile"; then
		fail=1; died=1; reason="one or more GPU workers DIED"
	fi
	if grep -qaE 'No clients are alive' "$logfile"; then
		fail=1; died=1; reason="all GPU workers died"
	fi
	if grep -qaE 'GPU [0-9]+: FAULTY' "$logfile"; then
		fail=1; reason="${reason:+$reason; }GPU(s) reported FAULTY (compute errors)"
	fi
	if grep -qaiE "couldn't init a GPU test|Error in load module|unsupported PTX|unsupported toolchain" "$logfile"; then
		fail=1; reason="${reason:+$reason; }a GPU test failed to initialize"
	fi
	if [ "$rc" -ne 0 ]; then
		fail=1; reason="${reason:+$reason; }gpu_burn exit code $rc"
	fi
	# A clean run must reach the end-of-test summary; its absence means it aborted.
	if ! grep -qaE '^Tested [0-9]+ GPUs:' "$logfile"; then
		fail=1; reason="${reason:+$reason; }no completion summary (gpu_burn aborted early)"
	fi

	rm -f "$logfile"
	if [ "$fail" -ne 0 ]; then
		echo "[gpu-burn] FAILED: $reason"
		[ "$died" -eq 1 ] && gpuburn_report_busy_gpus
		echo "===GPU Stress Test Failed==="
		return 1
	fi
	echo "===GPU Stress Test Success==="
	return 0
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
