#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
#apt-get install -y jq

$SCRIPT_DIR/ubimonitor -t -q 10 2>&1 | tee ib0_result.log

COUNT_ERR=$(cat ib0_result.log |grep ERROR | wc -l)

if [ $? -eq 0 ] && [ $COUNT_ERR == 0 ]; then

        echo "===IB Stress Test Success==="
else
        echo "===IB Stress Test Failed==="
fi

rm ib0_result.log
