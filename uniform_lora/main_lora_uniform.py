import sys
import argparse
import datetime
import numpy as np
import time
import torch

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from datasets import build_continual_dataloader
from engine import train_and_evaluate, evaluate_till_now
import utils
import os
import wandb

import warnings

warnings.filterwarnings("ignore", "Argument interpolation should be of type InterpolationMode instead of int")


def create_lora_model(args):
    """Create advanced LoRA model with shared adaptation and optional prototype classification."""
    print(f"Creating Advanced Shared LoRA model: {args.model}")
    
    # Create standard ViT backbone
    print(f"Loading pretrained weights: {args.pretrained}")
    vit_model = create_model(
        'vit_base_patch16_224',
        pretrained=args.pretrained,
        num_classes=0,  # No classification head in backbone
        drop_rate=getattr(args, 'drop', 0.0),
        drop_path_rate=getattr(args, 'drop_path', 0.0),
    )
    
    if args.pretrained:
        print("Successfully loaded pretrained ViT-B/16 weights")
    else:
        print("Using random initialization (no pretrained weights)")
    
    # Dataset-specific parameters
    if args.nb_classes == 35:  # clothing16k
        attr_num, obj_num, com_num = 9, 8, 35
    elif args.nb_classes == 80:  # ut-zappos
        attr_num, obj_num, com_num = 15, 12, 80
    elif args.nb_classes == 100:  # cgqa
        attr_num, obj_num, com_num = 31, 31, 100
    else:
        attr_num, obj_num, com_num = args.attr_num, args.obj_num, args.nb_classes
    
    # Import the TriLoRA model class
    from lora import CompILerTriLoRAViTModel, print_lora_info
    
    def _parse_layer_spec(spec: str, depth: int):
        """
        Spec grammar: "a:b,c,d:e,..."
        - a:b means [a, b) (start-inclusive, end-exclusive)
        - single ints allowed
        - negative indices allowed (relative to depth)
        """
        idxs = set()
        if not spec:
            return idxs
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                s, e = part.split(":")
                s = int(s); e = int(e)
                if s < 0: s += depth
                if e < 0: e += depth
                s = max(0, s); e = min(depth, e)
                if s < e:
                    idxs.update(range(s, e))
            else:
                i = int(part)
                if i < 0: i += depth
                if 0 <= i < depth:
                    idxs.add(i)
        return idxs

    # Build TriLoRA model
    depth = len(getattr(vit_model, "blocks", []))
    
    # Layer assignment logic
    # If any layer argument is provided (even if empty string), we use manual assignment.
    # We check if the argument is present in args to distinguish between "not provided" and "provided as empty string".
    # However, since argparse sets defaults, we check if any of them are non-default or explicitly meant to be set.
    # A safer check is: if ALL are empty/default, use auto. If ANY is set (e.g. comp_layers="0:12"), parse them all.
    
    # Check if user provided explicit layer configuration
    # Note: getattr(args, "attr_layers", "") returns "" if not set.
    # Ideally, the run script sets these.
    
    attr_spec = getattr(args, "attr_layers", "")
    obj_spec  = getattr(args, "obj_layers", "")
    comp_spec = getattr(args, "comp_layers", "")

    # Heuristic: if comp_spec is set (e.g. "0:12"), we assume manual mode.
    # Or if attr_spec/obj_spec are explicitly passed (even as empty strings in the run script).
    # To support the "uniform" case where attr/obj are empty, we should default to manual 
    # if comp_spec is non-empty or if we detect we are in a specific mode.
    
    if comp_spec or attr_spec or obj_spec:
        print(f"Manual Layer Assignment: Attr='{attr_spec}', Obj='{obj_spec}', Comp='{comp_spec}'")
        depth_assign = {
            "attr": _parse_layer_spec(attr_spec, depth),
            "obj":  _parse_layer_spec(obj_spec, depth),
            "comp": _parse_layer_spec(comp_spec, depth),
        }
    else:
        print("Auto Layer Assignment (LSA default)")
        depth_assign = "auto"

    ranks  = {"attr": getattr(args, 'lora_rank_attr', 8),  "obj": getattr(args, 'lora_rank_obj', 8),  "comp": getattr(args, 'lora_rank_comp', 8)}
    alphas = {"attr": getattr(args, 'lora_alpha_attr', 16.0), "obj": getattr(args, 'lora_alpha_obj', 16.0), "comp": getattr(args, 'lora_alpha_comp', 16.0)}

    model = CompILerTriLoRAViTModel(
        vit_model=vit_model,
        attr_num=attr_num, obj_num=obj_num, com_num=com_num,
        ranks=ranks, alphas=alphas, assign=depth_assign,
        use_proto_head=getattr(args, 'use_proto_head', False),
        proto_tau=getattr(args, 'proto_temp', 20.0)
    )
    
    # In Uniform mode, depth_assign['attr'] and ['obj'] will be empty sets.
    # CompILerTriLoRAViTModel will handle this via the updated lora.py logic.
    
    if depth_assign != "auto":
        print(f"  Attr layers: {sorted(list(depth_assign['attr']))}")
        print(f"  Obj layers:  {sorted(list(depth_assign['obj']))}")
        print(f"  Comp layers: {sorted(list(depth_assign['comp']))}")
    else:
        print("Using Tri-LoRA specialization (attr/obj/comp) with AUTO split")
    
    # Verify only LoRA + heads are trainable
    model.verify_frozen_backbone()
    print_lora_info(model)
    
    return model


def main(args):
    utils.init_distributed_mode(args)
    utils.fix_seed(args.seed)

    device = torch.device(args.device)
    data_loader, class_mask = build_continual_dataloader(args)

    # Create LoRA model
    model = create_lora_model(args)
    model.to(device)

    if args.freeze:
        # For LoRA model, freezing is handled automatically in the model
        # But we can still apply additional freezing if specified
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)) and 'lora' not in n and 'head' not in n:
                p.requires_grad = False

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        attr_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        obj_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, "checkpoint/task{}_checkpoint.pth".format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print("Loading checkpoint from:", checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model"])
            else:
                print("No checkpoint found at:", checkpoint_path)
                return
            _ = evaluate_till_now(
                model,
                data_loader,
                device,
                task_id,
                class_mask,
                acc_matrix,
                attr_acc_matrix,
                obj_acc_matrix,
                args,
            )

        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Print LoRA parameter info (already handled by print_lora_info in create_lora_model)

    # if args.unscale_lr:
    #     global_batch_size = args.batch_size
    # else:
    #     global_batch_size = args.batch_size * args.world_size
    # args.lr = args.lr * global_batch_size / 256.0

    args.lr = args.lr  

    # Loss functions - using standard CrossEntropy
    # Dataset-specific parameters
    if args.nb_classes == 35:  # clothing16k
        com_num = 35
    elif args.nb_classes == 80:  # ut-zappos
        com_num = 80
    elif args.nb_classes == 100:  # cgqa
        com_num = 100
    else:
        com_num = args.nb_classes
    
    # Criterion is now handled directly in engine.py with nn.CrossEntropyLoss
    criterion = [
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss()  # Second criterion for compatibility
    ]

    # Calculate learning rates based on multipliers
    base_lr = args.lr
    attr_lr = base_lr * getattr(args, 'attr_lr_multiplier', 1.0)
    head_lr = base_lr * getattr(args, 'head_lr_multiplier', 1.0)
    
    # Get parameter groups with different learning rates
    param_groups = model_without_ddp.get_parameter_groups(
        lora_lr=base_lr,
        head_lr=head_lr,
        attr_head_lr=attr_lr,
        lora_attr_mult=getattr(args, 'lora_attr_lr_multiplier', 1.0),
        lora_obj_mult=getattr(args, 'lora_obj_lr_multiplier', 1.0),
        lora_comp_mult=getattr(args, 'lora_comp_lr_multiplier', 1.0)
    )
    
    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=getattr(args, 'weight_decay', 0.01)
    )
    
    # Show detailed Tri-LoRA learning rates
    attr_mult = getattr(args, 'lora_attr_lr_multiplier', 1.0)
    obj_mult = getattr(args, 'lora_obj_lr_multiplier', 1.0)
    comp_mult = getattr(args, 'lora_comp_lr_multiplier', 1.0)
    print(f"Using Tri-LoRA separate learning rates:")
    print(f"  Attr LoRA: {base_lr * attr_mult} ({attr_mult}x)")
    print(f"  Obj LoRA: {base_lr * obj_mult} ({obj_mult}x)")
    print(f"  Comp LoRA: {base_lr * comp_mult} ({comp_mult}x)")
    print(f"  Attribute head: {attr_lr} ({getattr(args, 'attr_lr_multiplier', 1.0)}x)")
    print(f"  Other heads: {head_lr} ({getattr(args, 'head_lr_multiplier', 1.0)}x)")
    print(f"  Weight decay: {getattr(args, 'weight_decay', 0.01)}")

    # if args.sched != "constant":
    #     lr_scheduler, _ = create_scheduler(args, optimizer)
    # elif args.sched == "constant":
    #     lr_scheduler = None

    if getattr(args, "sched", "constant") != "constant":
        lr_scheduler, _ = create_scheduler(args, optimizer)
    else:
        lr_scheduler = None

    print(f"Start LoRA training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(
        model,
        model_without_ddp,
        criterion,
        data_loader,
        optimizer,
        lr_scheduler,
        device,
        class_mask,
        args,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CompILer LoRA training and evaluation configs")
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest="subparser_name")

    if config == "ut_zappos":
        from configs.ut_zappos_lora import get_args_parser

        config_parser = subparser.add_parser("ut_zappos", help="UT-Zappos LoRA configs")
    elif config == "clothing16k":
        from configs.clothing16k_lora import get_args_parser

        config_parser = subparser.add_parser("clothing16k", help="Clothing16k LoRA configs")
    elif config == "cgqa":
        from configs.cgqa_lora import get_args_parser

        config_parser = subparser.add_parser("cgqa", help="CGQA LoRA configs")
    else:
        raise NotImplementedError(f"{config} not found in configs/")

    get_args_parser(config_parser)
    
    # Add Tri-LoRA arguments
    config_parser.add_argument('--attr_layers', default="", type=str, help='Attr branch layers, e.g. "0:4"')
    config_parser.add_argument('--obj_layers',  default="", type=str, help='Obj  branch layers, e.g. "4:10"')
    config_parser.add_argument('--comp_layers', default="", type=str, help='Comp branch layers, e.g. "10:12"')
    config_parser.add_argument('--lora_rank_attr',  default=8,  type=int)
    config_parser.add_argument('--lora_rank_obj',   default=8,  type=int)
    config_parser.add_argument('--lora_rank_comp',  default=8,  type=int)
    config_parser.add_argument('--lora_alpha_attr', default=16.0, type=float)
    config_parser.add_argument('--lora_alpha_obj',  default=16.0, type=float)
    config_parser.add_argument('--lora_alpha_comp', default=16.0, type=float)
    config_parser.add_argument('--lora_attr_lr_multiplier', default=1.0, type=float)
    config_parser.add_argument('--lora_obj_lr_multiplier',  default=1.0, type=float)
    config_parser.add_argument('--lora_comp_lr_multiplier', default=1.0, type=float)
    # Per-branch loss weights
    config_parser.add_argument('--lambda_comp', default=0.5, type=float, help='Composition branch loss weight')
    config_parser.add_argument('--lambda_attr', default=0.25, type=float, help='Attribute branch loss weight')
    config_parser.add_argument('--lambda_obj', default=0.25, type=float, help='Object branch loss weight')
    config_parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")

    # Add prototype-related arguments
    config_parser.add_argument('--use_proto_head', default=False, action='store_true',
                              help='Use prototype-based composition head instead of linear')
    config_parser.add_argument('--proto_temp', default=20.0, type=float,
                              help='Temperature for cosine prototype logits')
    config_parser.add_argument('--proto_from', default='seen_train', type=str,
                              choices=['seen_train', 'seen_val'],
                              help='Which splits to build prototypes from at end of each task')

    # Add PGWS blending arguments
    config_parser.add_argument('--pgws_proto_blend', default=True, action='store_true',
                              help='Enable PGWS blending into prototypes')
    config_parser.add_argument('--pgws_eta_both', default=0.20, type=float,
                              help='PGWS blending factor when both primitives seen before')
    config_parser.add_argument('--pgws_eta_one', default=0.10, type=float,
                              help='PGWS blending factor when exactly one primitive seen before')
    config_parser.add_argument('--pgws_eta_none', default=0.00, type=float,
                              help='PGWS blending factor when no primitives seen before')

    config_parser.add_argument('--per_task_adapters', action='store_true', default=True,
                              help='Allocate a separate LoRA adapter for each task')
    config_parser.add_argument('--transport_blend', type=float, default=1.0,
                              help='EMA blend when merging transported prototypes into current matrix (1.0 = overwrite)')
    config_parser.add_argument('--transport_max_pairs', type=int, default=4096,
                              help='Max paired features to fit affine transport per old task')
    config_parser.add_argument('--track_drift', action='store_true', default=False,
                          help='Track and save prototype drift metrics for analysis')
    config_parser.add_argument('--drift_log_path', type=str, default=None,
                            help='Path to save drift metrics')
    config_parser.add_argument('--transport_ridge', type=float, default=1e-3,
                           help='L2 ridge λ for affine transport fit (0 disables bias penalty only row)')
    
    # EASE-style ensemble inference
    config_parser.add_argument('--ensemble_alpha', type=float, default=0.1,
                              help='Weight for off-diagonal in EASE ensemble (0.1 = EASE default)')


    args = parser.parse_args()
    

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Update output directory to distinguish LoRA runs
    if args.output_dir and not args.output_dir.endswith('_lora'):
        args.output_dir = args.output_dir + '_lora'
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb in offline mode for testing
    wandb.init(
        mode="offline",
        project="compiler-lora",
        config=vars(args),
        name=utils.get_run_name(args) + "_lora",
        tags=[
            f"{args.dataset}",
            f"{args.num_tasks}tasks",
            "shared_lora",
            f"rank{getattr(args, 'lora_rank', 8)}",
            "prototype" if getattr(args, 'use_proto_head', False) else "linear",
        ],
    )

    main(args)
    sys.exit(0)