#!/bin/bash

cat /etc/os-release |grep PRETTY_NAME > test.log

OS_TYPE=$(sed -n '1{s/^\(.............\)\(......\).*/\2/p}' test.log)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

rm test.log

if [ $OS_TYPE = "Ubuntu" ]; then
	dpkg -i $SCRIPT_DIR/memtester_4.3.0-4_amd64.deb
#	apt-get install -y jq	
	if [ $? -eq 0 ]; then
		echo "memtester installed success."
	else 
		echo "memtester installed failed."
	fi

elif [ $OS_TYPE = "CentOS" ]; then 
	yum install -y memtester
#	yum install -y jq
        if [ $? -eq 0 ]; then
                echo "memtester installed success."
        else
                echo "memtester installed failed."
        fi

else
	echo "Unsupported OS type: $OS_TYPE"
	exit 1
fi

#memtester 8GB 1
bash $SCRIPT_DIR/memtester_loop.sh 88 32 65536

if [ $? -eq 0 ]; then
        echo "===Memory Stress Test Success==="
else
        echo "===Memory Stress Test Failed==="
fi
