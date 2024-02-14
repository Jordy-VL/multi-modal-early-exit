#!/usr/bin/env bash

#define list of mdels here, space seperated, note the naming conventions
n_params=("Omar95farag/2024-01-11_one_stage_subgraphs_entropyreg_txt_vision_enc_all_ramp")
n_params2=("2024-01-09_one_stage_subgraphs_weighted_txt_vision_enc_all_ramp-rvl_cdip_100_examples_per_class")

for i in $(seq 0 $((${#n_params[@]} - 1)))
do
  echo "${n_params[$i]}"
  # Run the python command with the corresponding -n and exits parameters
  python3 eval.py -c "${n_params[$i]}" -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.05 --step 0.05 --exit_policy "max_confidence_global_thresholding_policy" --full_test True
  python3 eval.py -c "${n_params[$i]}" -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.05 --step 0.05 --exit_policy "max_confidence_global_thresholding_policy" --calibrate True --full_test True
  python3 eval.py -c "${n_params[$i]}" -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.05 --step 0.05 --exit_policy "accuracy_calibration_heuristic" --calibrate True --full_test True
  python large_scale.py --path /home/omar/workspace/EE/EE/results/"${n_params2[$i]}/"
done
