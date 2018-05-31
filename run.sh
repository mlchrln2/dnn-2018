#!/bin/bash
cd data-05-31-2018
python format_full_data.py
cd ../dnn
python nt3_baseline_keras2.py
cd ../results
jupyter notebook DeepLift_K2_version_test.ipynb
