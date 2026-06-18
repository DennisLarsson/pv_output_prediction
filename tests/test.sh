#!/bin/bash

python3 optimize_model.py tests/test_data.csv -n 2 --seed 42 -T 1 -o fitted_best_model.joblib

python3 predict_data.py tests/test_smhi.txt -m fitted_best_model.joblib -o predicted_pv_output_with_time.csv

diff tests/test_pred.csv predicted_pv_output_with_time.csv
