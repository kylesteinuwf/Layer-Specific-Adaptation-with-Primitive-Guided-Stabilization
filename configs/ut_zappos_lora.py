def get_args_parser(subparsers):
  
    # ============================================================================
    # CORE TRAINING PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--batch-size", default=16, type=int, 
                          help="Batch size per device")
    subparsers.add_argument("--epochs", default=5, type=int,
                          help="Number of epochs per task")
    
    # ============================================================================
    # MODEL PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--model", default="vit_base_patch16_224", type=str,
                          metavar="MODEL", help="Name of model to train")
    subparsers.add_argument("--input-size", default=224, type=int, 
                          help="Input image size")
    subparsers.add_argument("--pretrained", default=True, type=bool,
                          help="Load pretrained model weights")
    subparsers.add_argument("--drop", type=float, default=0.0,
                          help="Dropout rate")
    subparsers.add_argument("--drop-path", type=float, default=0.0,
                          help="Drop path rate") 
    
    # ============================================================================
    # OPTIMIZER PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--opt", default="adam", type=str,
                          help='Optimizer type')
    subparsers.add_argument("--lr", type=float, default=0.0001,
                          help="Base learning rate for LoRA parameters")
    subparsers.add_argument("--weight-decay", type=float, default=0.00,
                          help="Weight decay (0.01 for LoRA regularization)")
    subparsers.add_argument("--clip-grad", type=float, default=1.0,
                          help="Gradient clipping norm")
    subparsers.add_argument("--reinit_optimizer", type=bool, default=False,
                          help="Reinitialize optimizer for each task")
    
    # Learning rate multipliers for different parameter groups
    subparsers.add_argument("--attr_lr_multiplier", type=float, default=10.0,
                          help="Attribute head LR multiplier (vs base LR) - higher for challenging attributes")
    subparsers.add_argument("--head_lr_multiplier", type=float, default=5.0,
                          help="Other heads LR multiplier (vs base LR) - medium for composition/object heads")
    
    # ============================================================================
    # LEARNING RATE SCHEDULING 
    # ============================================================================  
    subparsers.add_argument("--sched", default="constant", type=str,
                          help='LR scheduler type')
    subparsers.add_argument("--unscale_lr", type=bool, default=True,
                          help="Apply learning rate scaling by batch size")
    
    # ============================================================================
    # LOSS FUNCTION PARAMETERS 
    # ============================================================================
  
    subparsers.add_argument("--alpha", default=0.1, type=float,
                          help="Primitive loss weight: L_total = L_comp + α*(L_attr + L_obj)")
    subparsers.add_argument("--beta", default=0.1, type=float,
                          help="Final score weight: score = logits + β*(attr + obj)")
    
    # ============================================================================
    # DATASET PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--data-path", default="./local_datasets/", type=str,
                          help="Path to datasets")
    subparsers.add_argument("--dataset", default="czsl-ut-zappos", type=str,
                          help="Dataset name")
    subparsers.add_argument("--nb_classes", default=80, type=int,
                          help="Total number of compositional classes")
    
    # ============================================================================
    # CONTINUAL LEARNING PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--num_tasks", default=5, type=int,
                          help="Number of sequential tasks")
    subparsers.add_argument("--train_mask", default=True, type=bool,
                          help="Apply class masking during training")
    subparsers.add_argument("--task_inc", default=False, type=bool,
                          help="Task-incremental evaluation (vs class-incremental)")
    
    # ============================================================================
    # SYSTEM PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--device", default="cuda", type=str,
                          help="Device for training")
    subparsers.add_argument("--seed", default=42, type=int,
                          help="Random seed for reproducibility")
    subparsers.add_argument("--output_dir", default="./output/ut_zappos", type=str,
                          help="Output directory for checkpoints and logs")
    
    # ============================================================================
    # DATA LOADING PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--num_workers", default=4, type=int, 
                          help="Number of data loading workers")
    subparsers.add_argument("--pin-mem", default=True, type=bool,
                          help="Pin memory in DataLoader")
    
    # ============================================================================
    # DISTRIBUTED TRAINING 
    # ============================================================================
    subparsers.add_argument("--world_size", default=1, type=int,
                          help="Number of distributed processes")
    subparsers.add_argument("--distributed", default=False, type=bool,
                          help="Enable distributed training")
    
    # ============================================================================
    # EVALUATION PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--eval", action="store_true",
                          help="Evaluation only mode")
    subparsers.add_argument("--print_freq", type=int, default=10,
                          help="Print frequency during training")
    
    # ============================================================================
    # OPTIONAL PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--freeze", default=[], nargs="*", type=str,
                          help="Additional parameters to freeze (optional, LoRA handles most freezing)")
    
 