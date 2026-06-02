#!/bin/bash

cat /etc/os-release |grep PRETTY_NAME > test.log

OS_TYPE=$(sed -n '1{s/^\(.............\)\(......\).*/\2/p}' test.log)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
rm test.log

if [ $OS_TYPE = "Ubuntu" ]; then
	dpkg -i $SCRIPT_DIR/*.deb
#	apt-get install -y jq
	if [ $? -eq 0 ]; then
		echo "stress-ng installed success."
	else 
		echo "stress-ng installed failed."
	fi

elif [ $OS_TYPE = "CentOS" ]; then 
	yum install -y stress-ng
#	yum install -y jq
        if [ $? -eq 0 ]; then
                echo "stress-ng installed success."
        else
                echo "stress-ng installed failed."
        fi

else
	echo "Unsupported OS type: $OS_TYPE"
	exit 1
fi

stress-ng --matrix 0 -t 10m --times --tz --metrics

if [ $? -eq 0 ]; then
        echo "===CPU Stress Test Success==="
else
        echo "===CPU Stress Test Failed==="
fi
