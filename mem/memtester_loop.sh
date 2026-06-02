#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   sudo bash memtester_loop.sh [覆盖率] [并发数] [预留MB]
# 例子:
#   sudo bash memtester_loop.sh 88 8 65536

TARGET_PERCENT=${1:-88}
WORKERS=${2:-8}
RESERVE_MB=${3:-65536}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "[ERROR] missing command: $1"
        exit 1
    }
}

need_cmd awk
need_cmd memtester

log() {
    echo "[$(date '+%F %T')] $*"
}

get_mem_available_mb() {
    awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo
}

get_mem_total_mb() {
    awk '/MemTotal:/ {print int($2/1024)}' /proc/meminfo
}

main() {
    local total_mb avail_mb target_mb worker_mb
    total_mb=$(get_mem_total_mb)
    avail_mb=$(get_mem_available_mb)

    if (( WORKERS <= 0 )); then
        echo "[ERROR] invalid WORKERS=${WORKERS}"
        exit 1
    fi

    target_mb=$(( avail_mb * TARGET_PERCENT / 100 - RESERVE_MB ))

    if (( target_mb < 1024 )); then
        echo "[ERROR] target memory too small: ${target_mb}MB"
        exit 1
    fi

    worker_mb=$(( target_mb / WORKERS ))

    if (( worker_mb < 512 )); then
        echo "[ERROR] each worker memory too small: ${worker_mb}MB"
        exit 1
    fi

    log "========== memory stress start =========="
    log "MemTotal     : ${total_mb} MB"
    log "MemAvailable : ${avail_mb} MB"
    log "TargetPercent: ${TARGET_PERCENT}%"
    log "Workers      : ${WORKERS}"
    log "Reserve      : ${RESERVE_MB} MB"
    log "Target       : ${target_mb} MB"
    log "Each worker  : ${worker_mb} MB"

    local pids=()
    local rc=0

    for ((i=0; i<WORKERS; i++)); do
        (
            echo "[worker-$i] start memtester ${worker_mb}M x1"
            memtester "${worker_mb}M" 1
            echo "[worker-$i] done"
        ) &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done

    log "========== memory stress end =========="

    if (( rc == 0 )); then
        echo "===Memory Stress Test Success==="
        exit 0
    else
        echo "===Memory Stress Test Failed==="
        exit 1
    fi
}

main
