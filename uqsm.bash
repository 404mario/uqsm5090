#!/bin/bash
# uqsm5090 - stress-test harness derived from uqsm v1.1.
# Adds --gpu-tool {dcgm|gpuburn|auto} so the GPU module can run on cards
# (e.g. RTX 5090 / Blackwell) where dcgmi diag does not yet work.

set -u

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$SCRIPT_DIR"

GPU_TOOL="auto"          # auto | dcgm | gpuburn
GPU_DURATION=600         # seconds, passed to gpu-burn
MODULE=""                # cpu | gpu | mem | ib (empty = all)

usage() {
	cat <<EOF
Usage: ./uqsm.bash [options]

Options:
  -dt <module>             Run a single module: cpu | gpu | mem | ib
                           (omit to run all four)
  --gpu-tool <tool>        GPU backend: dcgm | gpuburn | auto (default: auto)
                           auto = gpuburn if any RTX 5090 detected, else dcgm
  --gpu-duration <sec>     gpu-burn run length in seconds (default: 600)
  -h, --help               Show this help

Examples:
  ./uqsm.bash                                    # run all modules, auto GPU tool
  ./uqsm.bash -dt gpu                            # only GPU, auto tool
  ./uqsm.bash -dt gpu --gpu-tool gpuburn         # force gpu-burn
  ./uqsm.bash -dt gpu --gpu-tool dcgm            # force dcgmi diag -r 4
  ./uqsm.bash --gpu-tool gpuburn --gpu-duration 1800
EOF
}

# ---- arg parsing ---------------------------------------------------------
while [ $# -gt 0 ]; do
	case "$1" in
		-dt)
			MODULE="${2:-}"; shift 2 ;;
		--gpu-tool)
			GPU_TOOL="${2:-}"; shift 2 ;;
		--gpu-tool=*)
			GPU_TOOL="${1#*=}"; shift ;;
		--gpu-duration)
			GPU_DURATION="${2:-}"; shift 2 ;;
		--gpu-duration=*)
			GPU_DURATION="${1#*=}"; shift ;;
		-h|--help)
			usage; exit 0 ;;
		*)
			echo "Unknown argument: $1"; usage; exit 1 ;;
	esac
done

case "$GPU_TOOL" in
	auto|dcgm|gpuburn) ;;
	*) echo "Invalid --gpu-tool: $GPU_TOOL (expected dcgm|gpuburn|auto)"; exit 1 ;;
esac

# ---- gpu tool auto-detect ------------------------------------------------
resolve_gpu_tool() {
	if [ "$GPU_TOOL" != "auto" ]; then
		echo "$GPU_TOOL"; return
	fi
	# RTX 5090 / Blackwell: dcgmi diag is not supported yet -> prefer gpu-burn.
	if command -v nvidia-smi >/dev/null 2>&1 \
	   && nvidia-smi -L 2>/dev/null | grep -qiE 'RTX 5090|RTX 50[0-9]{2}|Blackwell'; then
		echo "gpuburn"
	else
		echo "dcgm"
	fi
}

# ---- json helpers --------------------------------------------------------
ensure_jq() {
	if ! command -v jq >/dev/null 2>&1; then
		if ls "$SCRIPT_DIR"/jq/*.deb >/dev/null 2>&1; then
			sudo dpkg -i "$SCRIPT_DIR"/jq/*.deb >/dev/null 2>&1 || \
				dpkg -i "$SCRIPT_DIR"/jq/*.deb >/dev/null 2>&1 || true
		fi
	fi
}

write_result() {
	# write_result <device> <result> <detail> <json_path>
	local device="$1" result="$2" detail="$3" path="$4"
	if command -v jq >/dev/null 2>&1; then
		jq -n --arg d "$device" --arg r "$result" --arg det "$detail" \
		      '{device:$d, result:$r, detail:$det}' > "$path"
	else
		# Minimal hand-rolled JSON fallback: escape \, ", and newlines.
		local esc
		esc=$(printf '%s' "$detail" \
			| awk 'BEGIN{ORS=""} {gsub(/\\/,"\\\\"); gsub(/"/,"\\\""); print; if(NR>0) print "\\n"}')
		printf '{"device":"%s","result":"%s","detail":"%s"}\n' \
			"$device" "$result" "$esc" > "$path"
	fi
}

# ---- module runners ------------------------------------------------------
stress_cpu() {
	local JSON_FILE="$SCRIPT_DIR/result/cpu_result.json"
	echo "Running CPU stress test..."
	"$SCRIPT_DIR/cpu/run_sm.bash" 2>&1 | tee cpu_result.log
	local last
	last=$(tail -n1 cpu_result.log)
	if [ "$last" = "===CPU Stress Test Success===" ]; then
		write_result "cpu" "pass" "" "$JSON_FILE"
	else
		write_result "cpu" "failed" "$(cat cpu_result.log)" "$JSON_FILE"
	fi
	rm -f cpu_result.log
}

stress_gpu() {
	local gpu_num
	gpu_num=$(lspci 2>/dev/null | grep -c NVIDIA)
	if [ "$gpu_num" -eq 0 ]; then
		echo "There is no GPU in the server."
		return
	fi
	local JSON_FILE="$SCRIPT_DIR/result/gpu_result.json"
	local tool; tool=$(resolve_gpu_tool)
	echo "Running GPU stress test (tool=$tool, duration=${GPU_DURATION}s)..."
	"$SCRIPT_DIR/gpu/run_sm.bash" "$tool" "$GPU_DURATION" 2>&1 | tee gpu_result.log
	local last
	last=$(tail -n1 gpu_result.log)
	if [ "$last" = "===GPU Stress Test Success===" ]; then
		write_result "gpu" "pass" "" "$JSON_FILE"
	else
		write_result "gpu" "failed" "$(cat gpu_result.log)" "$JSON_FILE"
	fi
	rm -f gpu_result.log
}

stress_mem() {
	local JSON_FILE="$SCRIPT_DIR/result/mem_result.json"
	echo "Running MEM stress test..."
	"$SCRIPT_DIR/mem/run_sm.bash" 2>&1 | tee mem_result.log
	local last
	last=$(tail -n1 mem_result.log)
	if [ "$last" = "===Memory Stress Test Success===" ]; then
		write_result "mem" "pass" "" "$JSON_FILE"
	else
		write_result "mem" "failed" "$(cat mem_result.log)" "$JSON_FILE"
	fi
	rm -f mem_result.log
}

stress_ib() {
	local ib_num
	ib_num=$(lspci 2>/dev/null | grep -c Infiniband)
	if [ "$ib_num" -eq 0 ]; then
		echo "There is no IB in the server."
		return
	fi
	local JSON_FILE="$SCRIPT_DIR/result/ib_result.json"
	echo "Running IB stress test..."
	"$SCRIPT_DIR/ib/run_sm.bash" 2>&1 | tee ib_result.log
	local last
	last=$(tail -n1 ib_result.log)
	if [ "$last" = "===IB Stress Test Success===" ]; then
		write_result "ib" "pass" "" "$JSON_FILE"
	else
		write_result "ib" "failed" "$(cat ib_result.log)" "$JSON_FILE"
	fi
	rm -f ib_result.log perftest_out.json 2>/dev/null || true
}

generate_json_report() {
	echo "Generating JSON report..."
	local out="$SCRIPT_DIR/result/report.json"
	local parts=()
	for m in cpu gpu mem ib; do
		local f="$SCRIPT_DIR/result/${m}_result.json"
		[ -f "$f" ] && parts+=("$f")
	done
	if command -v jq >/dev/null 2>&1 && [ ${#parts[@]} -gt 0 ]; then
		jq -s '.' "${parts[@]}" > "$out"
	else
		{
			echo "["
			local first=1
			for f in "${parts[@]}"; do
				[ $first -eq 1 ] || echo ","
				cat "$f"
				first=0
			done
			echo "]"
		} > "$out"
	fi
	echo "Report generated: $out"
}

# ---- entry ---------------------------------------------------------------
ensure_jq
rm -f "$SCRIPT_DIR"/result/*.json 2>/dev/null || true

if [ -z "$MODULE" ]; then
	stress_cpu
	stress_gpu
	stress_mem
	stress_ib
else
	case "$MODULE" in
		cpu) stress_cpu ;;
		gpu) stress_gpu ;;
		mem) stress_mem ;;
		ib)  stress_ib  ;;
		*) echo "Invalid module: $MODULE (expected cpu|gpu|mem|ib)"; exit 1 ;;
	esac
fi

generate_json_report
