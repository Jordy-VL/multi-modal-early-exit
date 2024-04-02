# Description: Train the full dataset with the best models configurations in the previous step
# Independent_Single_full
python3 IC_only.py with layoutlmv3 model=EElayoutlmv3 dataset=jordyvl/rvl_cdip_easyocr epochs=60 batch_size=2 gradient_accumulation_steps=24 gamma=0.7 encoder_layer_strategy=ramp training_strategy=one_stage_subgraphs_weighted exits="text_avg,vision_avg,7" -n Independent_Single_Full
# Concat_Quarter_Ramp_Full
python3 IC_only.py with layoutlmv3 model=EElayoutlmv3 dataset=jordyvl/rvl_cdip_easyocr epochs=60 batch_size=2 gradient_accumulation_steps=24 gamma=0.7 encoder_layer_strategy=ramp training_strategy=one_stage_subgraphs_weighted exits="text_visual_concat,1,4,8,10" -n Concat_Quarter_Ramp_Full
# Concat_Alternate_Ramp_Full
python3 IC_only.py with layoutlmv3 model=EElayoutlmv3 dataset=jordyvl/rvl_cdip_easyocr epochs=60 batch_size=2 gradient_accumulation_steps=24 gamma=0.7 encoder_layer_strategy=ramp training_strategy=one_stage_subgraphs_weighted exits="text_visual_concat,2,5,9,11" -n Concat_Alternate_Ramp_Full
# Independent_Quarter_Ramp_Full
python3 IC_only.py with layoutlmv3 model=EElayoutlmv3 dataset=jordyvl/rvl_cdip_easyocr epochs=60 batch_size=2 gradient_accumulation_steps=24 gamma=0.7 encoder_layer_strategy=ramp training_strategy=one_stage_subgraphs_weighted exits="text_avg,vision_avg,1,4,8,10" -n Independent_Quarter_Ramp_Full
