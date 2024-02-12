

## Possible installations required

- `!pip install -q datasets`
- `!sudo apt install tesseract-ocr`
- `!pip install -q pytesseract`

- ### To run experiment with a specific policy and exit threshold
  - use `--exit_threshold`
  - use `--exit_policy`
  
  `python3 eval.py -c jordyvl/outlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-12-01_txt_vis_concat_enc_5_6_7_8_gate -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.13  --exit_policy "max_confidence_global_thresholding_policy"`

  - To run with accuracy_ece heuristic
    - use `--epsilon`
  
  `python3 eval.py -c Omar95farag/EElayoutlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-09-05_txt_vis_con_enc_4_6_7_11_12_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.2 --step 0.1 --exit_policy "accuracy_calibration_heuristic" --calibrate True --epsilon 0.1`

- ### To run experiment with a specific policy and a sweep on thresholds
   - add `--exit_threshold`: start threshold
   - add `--step`
   - add `--exit_policy`
   - add `--full_test`
     
   `python3 eval.py -c Omar95farag/EElayoutlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-09-05_txt_vis_con_enc_4_6_7_11_12_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.3 --step 0.1 --full_test True --exit_policy "max_confidence_global_thresholding_policy"`

- ### To run any test with calibrated logits
  - add `calibrate`
 
  `python3 eval.py -c Omar95farag/EElayoutlmv3_jordyvl_rvl_cdip_100_examples_per_class_2023-09-05_txt_vis_con_enc_4_6_7_11_12_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.3 --step 0.1 --full_test True --calibrate True --exit_policy "max_confidence_global_thresholding_policy"`
     
