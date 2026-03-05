# Paper Title: 
LSA-PAST: Layer-Specific Adaptation and Prototype Transport for Compositional Incremental Learning


# Abstract:
Class-Incremental Learning (CIL) enables models to recognize new classes over time while retaining prior knowledge, but many real-world problems are compositional, where visual concepts arise from combinations of primitives such as states (i.e., red, shiny) and objects (i.e., shoe, car). Standard CIL methods treat each composition as an independent class, ignoring shared primitives and providing no mechanism to stabilize their representations as new compositions arrive. This causes parameter updates to entangle primitives and induce drift that degrades performance on earlier compositions.  Compositional Incremental Learning (composition-IL) formalizes this setting by introducing new state–object compositions across tasks. In this paper, we propose Layer-Specific Adaptation with Prototype-Anchored Adapter-Space Transport (\textbf{LSA-PAST}), a parameter-efficient method tailored to composition-IL. LSA integrates Low-Rank Adaptation (LoRA) into a frozen ViT with depth-specialized adapters for states, objects, and compositions. PAST then maps past, anchored prototypes into the current adapter space through a rehearsal-free affine transform fit on current-task data, enabling old prototypes to remain compatible with new adapters. Extensive experiments across multiple benchmarks demonstrates that LSA-PAST achieves state-of-the-art performance for composition-IL.

# Datasets
The method was ran on Split-UT-Zappos, Split-Clothing, and Split-C-GQA. The cleaned datasets can be found here at this google drive link. Attach the folder into the repo. 

# Running the Scripts

## Split-Clothing
```bash  
python main_lora.py clothing16k --lr 1e-4 --epochs 10 --num_tasks 5 --use_proto_head --proto_temp 20.0 --proto_from seen_train --beta 0 --lora_rank_attr 8 --lora_rank_obj 8 --lora_rank_comp 8 --lora_alpha_attr 16.0 --lora_alpha_obj 16.0 --lora_alpha_comp 16.0 --lambda_comp 1.0 --lambda_attr 0.1 --lambda_obj 0.1 --attr_lr_multiplier 10.0 --head_lr_multiplier 1.0 --attr_layers "0:3" --obj_layers "4:9" --comp_layers "10:12" --per_task_adapters --transport_max_pairs 4096 --pgws_eta_both 0.2 --pgws_eta_one 0.1 --ensemble_alpha 0.1 --seed 192
```

## Split-UT-Zappos
```bash  
python main_lora.py ut_zappos --lr 1e-4 --epochs 15 --num_tasks 5 --use_proto_head --proto_temp 20.0 --proto_from seen_train --beta 0 --lora_rank_attr 8 --lora_rank_obj 8 --lora_rank_comp 8 --lora_alpha_attr 16.0 --lora_alpha_obj 16.0 --lora_alpha_comp 16.0 --lambda_comp 1.0 --lambda_attr 0.1 --lambda_obj 0.1 --attr_lr_multiplier 10.0 --head_lr_multiplier 1.0 --attr_layers "0:5" --obj_layers "5:8" --comp_layers "8:12" --per_task_adapters --transport_max_pairs 10000 --pgws_eta_both 0.20 --pgws_eta_one 0.10 --ensemble_alpha 0.1 --seed 42 
```

## Split-C-GQA
```bash  
python main_lora.py cgqa --lr 1e-4 --epochs 10 --num_tasks 5 --use_proto_head --proto_temp 20.0 --proto_from seen_train --beta 0 --lora_rank_attr 8 --lora_rank_obj 8 --lora_rank_comp 8 --lora_alpha_attr 16.0 --lora_alpha_obj 16.0 --lora_alpha_comp 16.0 --lambda_comp 1.0 --lambda_attr 0.1 --lambda_obj 0.1 --attr_lr_multiplier 10.0 --head_lr_multiplier 1.0 --attr_layers "0:4" --obj_layers "4:9" --comp_layers "9:12" --per_task_adapters --transport_max_pairs 10000 --transport_blend 1.0 --pgws_eta_both 0.25 --pgws_eta_one 0.125 --ensemble_alpha 0.1 --seed 33
``` 

# Acknowledgement

We would like to thank the below repositories for their contributions. 

[CompILer](https://github.com/Yanyi-Zhang/CompILer) 

