def get_args_parser(subparsers):
    
    # ============================================================================
    # CORE TRAINING PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--batch-size", default=8, type=int, 
                          help="Batch size per device (reduced for CGQA's larger scale)")
    subparsers.add_argument("--epochs", default=3, type=int,
                          help="Number of epochs per task (reduced for large dataset)")
    
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
    subparsers.add_argument("--weight-decay", type=float, default=0.01,
                          help="Weight decay (0.01 for LoRA regularization)")
    subparsers.add_argument("--clip-grad", type=float, default=1.0,
                          help="Gradient clipping norm")
    subparsers.add_argument("--reinit_optimizer", type=bool, default=False,
                          help="Reinitialize optimizer for each task")
    
    # Learning rate multipliers for different parameter groups
    subparsers.add_argument("--attr_lr_multiplier", type=float, default=5.0,
                          help="Attribute head LR multiplier (vs base LR) - medium boost for color attributes")
    subparsers.add_argument("--head_lr_multiplier", type=float, default=3.0,
                          help="Other heads LR multiplier (vs base LR) - small boost for composition/object heads")
    
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
    subparsers.add_argument("--beta", default=0.0, type=float,
                          help="Final score weight: score = logits + β*(attr + obj)")
    
    # ============================================================================
    # DATASET PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--data-path", default="./local_datasets/", type=str,
                          help="Path to datasets")
    subparsers.add_argument("--dataset", default="czsl-cgqa", type=str,
                          help="Dataset name")
    subparsers.add_argument("--nb_classes", default=100, type=int,
                          help="Total number of compositional classes")
    subparsers.add_argument("--attr_num", default=31, type=int,
                          help="Number of attributes in CGQA")
    subparsers.add_argument("--obj_num", default=31, type=int,
                          help="Number of objects in CGQA")
    
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
    subparsers.add_argument("--output_dir", default="./output/clothing16k", type=str,
                          help="Output directory for checkpoints and logs")
    subparsers.add_argument("--warmup-epochs", type=int, default=1,
                        help="Number of warmup epochs for the LR scheduler")
    subparsers.add_argument("--warmup-lr", type=float, default=1e-6,
                            help="Starting LR during warmup")
    subparsers.add_argument("--min-lr", type=float, default=1e-6,
                            help="Lower LR bound for cosine/step schedulers")
    subparsers.add_argument("--cooldown-epochs", type=int, default=0,
                            help="Epochs to run at min-lr after the schedule ends")

    
    # ============================================================================
    # DATA LOADING PARAMETERS 
    # ============================================================================
    subparsers.add_argument("--num_workers", default=8, type=int,
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
    
