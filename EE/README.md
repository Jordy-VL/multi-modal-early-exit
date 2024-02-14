
## Training
- ### Important parameters
  - `gamma`: for weighing exit losses and top classifier
    - Options: values in range of [0,1]  
  - `encoder_layer_strategy`: output classifier type
    - Options: gate/ramp
  - `training_strategy`: choose training strategy for the EE design
    -  Options: one_stage_subgraphs_weighted/one_stage_subgraphs_entropyreg/one_stage_subgraphs_weighted_entropyreg"
  - `exits`: exit placement configuration
    - Options:
        - For modalities (one or more): text_avg/vision_avg/text_visual_concat (e.g, text_avg,vision_avg or text_visual_concat)
        - For encoders (one or more): values from 1 to 12 (e.g 1,5,9)
            
- `python3 IC_only.py with layoutlmv3 model=EElayoutlmv3 dataset=jordyvl/rvl_cdip_100_examples_per_class epochs=60 batch_size=2 gradient_accumulation_steps=24 gamma=0.7 encoder_layer_strategy=ramp training_strategy=one_stage_subgraphs_weighted exits="text_avg,vision_avg,7" -n Independent_Single`

## Testing

- ### Important parameters
  - `--exit_policy`: select the exit policy for inference
    - Options: "max_confidence_global_thresholding_policy", "accuracy_calibration_heuristic"
  - `--calibrate`: Set to True to run calibration
    - To disable calibration, make sure to remove the flag
    - Crucial to run with "accuracy_calibration_heuristic"
  - `--exit_threshold`: Set threshold
    - Options: values in range of [0,1]
  - `--full_test`: To run specific policy and with a sweep on thresholds (e.g max_confidence_global_thresholding_policy from 0 to 1)
  - `--step`: Use when `full_test` is set, to assign the threshods step for a sweep.
  - `--epsilon`: Set epsilon value for Min-Max normalization in "accuracy_calibration_heuristic"


### To run full test with all exit_polices
  - run `bash full_test.sh`
      - Check file for more details

- ### To run experiment with a specific policy and exit threshold
  - use `--exit_threshold`
  - use `--exit_policy`
  
  `python3 eval.py -c Omar95farag/2024-01-11_one_stage_subgraphs_entropyreg_txt_vision_enc_all_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.13  --exit_policy "max_confidence_global_thresholding_policy"`

  - To run with accuracy_ece heuristic
    - use `--epsilon`
  
  `python3 eval.py -c Omar95farag/2024-01-11_one_stage_subgraphs_entropyreg_txt_vision_enc_all_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.2 --step 0.1 --exit_policy "accuracy_calibration_heuristic" --calibrate True --epsilon 0.1`

- ### To run experiment with a specific policy and a sweep on thresholds
   - add `--exit_threshold`: start threshold
   - add `--step`
   - add `--exit_policy`
   - add `--full_test`
     
   `python3 eval.py -c Omar95farag/2024-01-11_one_stage_subgraphs_entropyreg_txt_vision_enc_all_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.3 --step 0.1 --full_test True --exit_policy "max_confidence_global_thresholding_policy"`

- ### To run any test with calibrated logits
  - add `calibrate`
 
  `python3 eval.py -c Omar95farag/2024-01-11_one_stage_subgraphs_entropyreg_txt_vision_enc_all_ramp -d jordyvl/rvl_cdip_100_examples_per_class --exit_threshold 0.3 --step 0.1 --full_test True --calibrate True --exit_policy "max_confidence_global_thresholding_policy"`
     
