#!/bin/bash
counter=$1
while [ $counter -gt 0 ]
do
	echo $counter
	cd data-05-31-2018
	python format_full_data.py
	cd ../dnn
	python nt3_baseline_keras2.py
	cd ../results
	python deeplift_test.py
	cd ../analysis
	python analyze.py
	cd ../
	counter=$(($counter - 1))
done
