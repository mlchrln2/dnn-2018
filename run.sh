#!/bin/bash
startval=$1
endval=$2
while [ $startval != $(($endval + 1)) ]
do
	progress="----------------------progress: step $startval of $endval---------------------"
	echo $progress
	cd data-05-31-2018
	#python format_full_data.py
	cd ../dnn
	#python nt3_baseline_keras2.py $startval
	cd ../results
	python deeplift_test.py $startval
	cd ../analysis
	python analyze.py $startval
	cd ../
	startval=$(($startval + 1))
done
