cat output/iterations.log | grep "$1" | while read -r line; do data=($line); echo ${data[$2]}; done
