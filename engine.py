import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import wandb

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from timm.utils import accuracy



import utils
from lora import (get_primitives_gt, 
                  build_prototypes_from_loaders, apply_compiler_masking_train,
                  apply_compiler_masking_eval)


@torch.no_grad()
def _fit_affine_transport(model, loader_cur_task, src_task_id: int, tgt_task_id: int, device, 
                          max_pairs=4096, ridge=1e-3):
    """
    Learn z_tgt ≈ A z_src + b using ONLY current task images by extracting paired
    features under src and tgt adapters.
    """
    Zs, Zt, n = [], [], 0
    model.eval()
    for xb, _ in loader_cur_task['train']:
        xb = xb.to(device, non_blocking=True)
        zs = model.extract_features(xb, task_id=src_task_id)   # old adapter
        zt = model.extract_features(xb, task_id=tgt_task_id)   # current adapter
        Zs.append(zs); Zt.append(zt)
        n += zs.shape[0]
        if n >= max_pairs: break

    Zs = torch.cat(Zs, 0)[:max_pairs]  # [N,D]
    Zt = torch.cat(Zt, 0)[:max_pairs]  # [N,D]

    # Sanity check for identical features (adapter routing issue)
    raw_cos = torch.nn.functional.cosine_similarity(Zs, Zt, dim=1).mean().item()
    print(f"[Sanity] src={src_task_id} tgt={tgt_task_id} mean cos(raw)={raw_cos:.6f}")
    

    N, D = Zs.shape
    ones = torch.ones(N, 1, device=Zs.device, dtype=Zs.dtype)
    X = torch.cat([Zs, ones], dim=1)           # [N, D+1]
    I = torch.eye(D+1, device=X.device, dtype=X.dtype)
    I[-1, -1] = 0.0  # don't regularize bias

    # (X^T X + λI) W = X^T Y;  W ∈ R^{(D+1)×D}
    W = torch.linalg.solve(X.T @ X + ridge * I, X.T @ Zt)
    A = W[:-1, :].T   # [D,D]
    b = W[-1, :]      # [D] - no transpose needed for 1D
    
    # Sanity check logging
    with torch.no_grad():
        pred = (Zs @ A.T) + b
        cos = (pred * Zt).sum(dim=1) / (pred.norm(dim=1) * Zt.norm(dim=1) + 1e-8)
        print(f"[Transport {src_task_id}->{tgt_task_id}] mean cos={cos.mean().item():.4f} (N={Zs.size(0)})")
    
    return A.contiguous(), b.contiguous()



def train_one_epoch(
    model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    set_training_mode=True,
    task_id=-1,
    class_mask=None,
    args=None,
):
    model.train(set_training_mode)

    if args.dataset == "czsl-clothing16k":
        dataset = "clothing16k"
        attr_num = 9
        obj_num = 8
        com_num = 35
    elif args.dataset == "czsl-ut-zappos":
        dataset = "ut-zappos"
        attr_num = 15
        obj_num = 12
        com_num = 80
    elif args.dataset == "czsl-cgqa":
        dataset = "cgqa"
        attr_num = 31
        obj_num = 31
        com_num = 100


    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    device = torch.device(args.device)
    
    # Standard CrossEntropy loss for all three heads
    CE_com = nn.CrossEntropyLoss()
    CE_attr = nn.CrossEntropyLoss()
    CE_obj = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("Loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"Train: [Task {task_id+1}] Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]"

    batch_count = 0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        batch_count += 1
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, task_id=task_id, train=set_training_mode)

        logits = output["com_logits"]
        attr_logits = output["attr_logits"]
        obj_logits = output["obj_logits"]

        # Apply CompILer's exact training masking logic using dedicated function
        logits, attr_logits, obj_logits = apply_compiler_masking_train(
            logits, attr_logits, obj_logits, dataset, task_id, class_mask, 
            attr_num, obj_num, args.nb_classes, args
        )

        # attr, obj gt and embddings
        # TODO Use attr_emb and obj_emb
        attribute_gt, object_gt = get_primitives_gt(dataset, target)
        com_loss = CE_com(logits, target)
        attr_loss = CE_attr(attr_logits, attribute_gt)
        obj_loss = CE_obj(obj_logits, object_gt)
        loss = getattr(args, 'lambda_comp', 0.5) * com_loss \
             + getattr(args, 'lambda_attr', 0.25) * attr_loss \
             + getattr(args, 'lambda_obj', 0.25) * obj_loss

        total_logits = logits  # No beta fusion - composition logits only
        acc1, acc5 = accuracy(total_logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            message = "Loss is {}, stopping training".format(loss.item())
            print(message)
            wandb.alert(message)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient sanity check for soft specialization
        def block_norm(model, lo, hi):
            s = 0.0
            for n,p in model.named_parameters():
                if any(f"blocks.{i}." in n for i in range(lo,hi)) and p.grad is not None:
                    s += (p.grad.detach()**2).sum().item()
            return (s + 1e-12)**0.5

        if batch_count % 50 == 0:  # Print every 50 iterations
            print("[GRAD] early:", f"{block_norm(model,0,4):.2e}",
                  "mid:", f"{block_norm(model,4,9):.2e}",
                  "late:", f"{block_norm(model,9,12):.2e}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
        metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

        wandb.log(
            {
                f"Task-{task_id}/Loss": loss.item(),
                f"Task-{task_id}/Lr": optimizer.param_groups[0]["lr"],
                f"Task-{task_id}/Acc@1": acc1.item(),
                f"Task-{task_id}/Acc@5": acc5.item(),
            }
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def ensemble_inference(model, images, eval_task_id, class_mask, args, alpha=0.1):
    device = images.device
    B = images.size(0)
    nb_classes = args.nb_classes
    tau = getattr(args, "proto_tau", getattr(args, "tau", 20.0))

    logits_full = torch.zeros(B, nb_classes, device=device)

    num_spaces = eval_task_id + 1  # adapters 0..eval_task_id
    for s in range(num_spaces):
        # features in space s
        out_s = model(images, task_id=s, train=False)
        z_s = out_s["pre_logits"]
        z_s = z_s / (z_s.norm(dim=1, keepdim=True) + 1e-8)

        # accumulate scores for each task block t in space s
        for t in range(num_spaces):
            key = (t, s)
            if not hasattr(model, "proto_matrix") or key not in model.proto_matrix:
                continue
            class_ids = torch.as_tensor(class_mask[t], device=device, dtype=torch.long)
            protos_ts = model.proto_matrix[key]  # [K_t, D] - already in correct order
            scores = tau * (z_s @ protos_ts.t())  # [B, K_t]
            w = 1.0 if (s == t) else alpha
            logits_full.index_add_(1, class_ids, w * scores)

    return logits_full



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device,
    class_mask=None,
    args=None,
    current_task_id=None,  # NEW: task we've trained up to
    test_task_id=None,     # NEW: task being tested
):
    if args.dataset == "czsl-clothing16k":
        dataset = "clothing16k"
        attr_num = 9
        obj_num = 8
        com_num = 35
    elif args.dataset == "czsl-ut-zappos":
        dataset = "ut-zappos"
        attr_num = 15
        obj_num = 12
        com_num = 80
    elif args.dataset == "czsl-cgqa":
        dataset = "cgqa"
        attr_num = 31
        obj_num = 31
        com_num = 100

    # Use standard CrossEntropy loss for all three heads
    CE_com = nn.CrossEntropyLoss()
    CE_attr = nn.CrossEntropyLoss()
    CE_obj = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    data_task_id = test_task_id if test_task_id is not None else current_task_id
    header = "Test: [Task {}]".format(data_task_id + 1)

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = ensemble_inference(
                model=model,
                images=input,
                eval_task_id=current_task_id,  # ✅ All adapters trained so far
                class_mask=class_mask,
                args=args,
                alpha=getattr(args, "ensemble_alpha", 0.1),
            )
    

            attr_logits = torch.zeros_like(logits)
            obj_logits  = torch.zeros_like(logits)

            attribute_gt, object_gt = get_primitives_gt(dataset, target)

            # Apply CompILer's exact evaluation masking logic using dedicated function
            logits, attr_logits, obj_logits = apply_compiler_masking_eval(
                logits, attr_logits, obj_logits, dataset, data_task_id, class_mask,
                attr_num, obj_num, args.nb_classes, args
            )

            com_loss = CE_com(logits, target)
            attr_loss = CE_attr(attr_logits, attribute_gt)
            obj_loss = CE_obj(obj_logits, object_gt)
            loss = getattr(args, 'lambda_comp', 0.5) * com_loss \
             + getattr(args, 'lambda_attr', 0.25) * attr_loss \
             + getattr(args, 'lambda_obj', 0.25) * obj_loss

            total_logits = logits  # No beta fusion - composition logits only
            acc1, acc5 = accuracy(total_logits, target, topk=(1, 5))

            attr_acc1, attr_acc5 = accuracy_primitives(total_logits, attribute_gt, dataset, "attr", topk=(1, 5))
            obj_acc1, obj_acc5 = accuracy_primitives(total_logits, object_gt, dataset, "obj", topk=(1, 5))

            # Compute harmonic mean of attribute and object accuracies 
            harmonic_mean_acc1 = compute_harmonic_mean(attr_acc1.item(), obj_acc1.item())
            harmonic_mean_acc5 = compute_harmonic_mean(attr_acc5.item(), obj_acc5.item())

            metric_logger.meters["Loss"].update(loss.item())
            metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
            metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

            metric_logger.meters["Attr_Acc@1"].update(attr_acc1.item(), n=input.shape[0])
            metric_logger.meters["Attr_Acc@5"].update(attr_acc5.item(), n=input.shape[0])
            metric_logger.meters["Obj_Acc@1"].update(obj_acc1.item(), n=input.shape[0])
            metric_logger.meters["Obj_Acc@5"].update(obj_acc5.item(), n=input.shape[0])
            metric_logger.meters["Harmonic_Mean@1"].update(harmonic_mean_acc1, n=input.shape[0])
            metric_logger.meters["Harmonic_Mean@5"].update(harmonic_mean_acc5, n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Attr_Acc@1 {attr_top1.global_avg:.3f} Obj_Acc@1 {obj_top1.global_avg:.3f} Harmonic_Mean@1 {hmean_top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.meters["Acc@1"],
            attr_top1=metric_logger.meters["Attr_Acc@1"],
            obj_top1=metric_logger.meters["Obj_Acc@1"],
            hmean_top1=metric_logger.meters["Harmonic_Mean@1"],
            losses=metric_logger.meters["Loss"],
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(
    model: torch.nn.Module,
    data_loader,
    device,
    task_id=-1,
    class_mask=None,
    acc_matrix=None,
    attr_acc_matrix=None,
    obj_acc_matrix=None,
    args=None,
):
    stat_matrix = np.zeros((5, args.num_tasks))  # Added slot for harmonic mean

    for i in range(task_id + 1):
        test_stats = evaluate(
            model=model,
            data_loader=data_loader[i]["val"],
            device=device,
            current_task_id=task_id,  # ✅ Use current training task
            test_task_id=i,           # Which val set (for logging)
            class_mask=class_mask,
            args=args
        )
        stat_matrix[0, i] = test_stats["Acc@1"]
        stat_matrix[1, i] = test_stats["Attr_Acc@1"]
        stat_matrix[2, i] = test_stats["Obj_Acc@1"]
        stat_matrix[3, i] = test_stats["Harmonic_Mean@1"]
        stat_matrix[4, i] = test_stats["Loss"]

        acc_matrix[i, task_id] = test_stats["Acc@1"]
        attr_acc_matrix[i, task_id] = test_stats["Attr_Acc@1"]
        obj_acc_matrix[i, task_id] = test_stats["Obj_Acc@1"]

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    diagonal = np.diag(acc_matrix)
    attr_diagonal = np.diag(attr_acc_matrix)
    obj_diagonal = np.diag(obj_acc_matrix)
    result_str = (
        "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAttr_Acc@1: {:.4f}\tObj_Acc@1: {:.4f}\tHarmonic_Mean@1: {:.4f}\tLoss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2], avg_stat[3], avg_stat[4]
        )
    )
    wandb.log(
        {
            "OVER ALL/Task ID": task_id,
            "OVER ALL/AVG Acc@1": avg_stat[0],
            "OVER ALL/AVG Attr_Acc@1": avg_stat[1],
            "OVER ALL/AVG Obj_Acc@1": avg_stat[2],
            "OVER ALL/AVG Harmonic_Mean@1": avg_stat[3],
            "OVER ALL/AVG Loss": avg_stat[4],
        }
    )
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        attr_forgetting = np.mean((np.max(attr_acc_matrix, axis=1) - attr_acc_matrix[:, task_id])[:task_id])
        attr_backward = np.mean((attr_acc_matrix[:, task_id] - attr_diagonal)[:task_id])

        obj_forgetting = np.mean((np.max(obj_acc_matrix, axis=1) - obj_acc_matrix[:, task_id])[:task_id])
        obj_backward = np.mean((obj_acc_matrix[:, task_id] - obj_diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}\tAttr_Forgetting: {:.4f}\tAttr_Backward: {:.4f}\tObj_Forgetting: {:.4f}\tObj_Backward: {:.4f}".format(
            forgetting, backward, attr_forgetting, attr_backward, obj_forgetting, obj_backward
        )

        wandb.log(
            {
                "OVER ALL/forgetting Task ID": task_id,
                "OVER ALL/forgetting": forgetting,
                "OVER ALL/backward": backward,
                "OVER ALL/attr_forgetting": attr_forgetting,
                "OVER ALL/attr_backward": attr_backward,
                "OVER ALL/obj_forgetting": obj_forgetting,
                "OVER ALL/obj_backward": obj_backward,
            }
        )
    print(result_str)

    return test_stats


def train_and_evaluate(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: torch.device,
    class_mask=None,
    args=None,
):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    attr_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    obj_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))


    for task_id in range(args.num_tasks):
        # Register new task and allocate fresh LoRA params
        model_without_ddp.register_new_task(task_id)
        model_without_ddp.set_active_task(task_id)

        # Add new task's LoRA params to existing optimizer (when reinit is OFF)
        if not args.reinit_optimizer:
            # Build fresh groups (this lists ALL trainable params, including the new task's LoRA)
            base_lr = args.lr
            attr_lr = base_lr * getattr(args, 'attr_lr_multiplier', 1.0)
            head_lr = base_lr * getattr(args, 'head_lr_multiplier', 1.0)
            fresh_groups = model_without_ddp.get_parameter_groups(
                lora_lr=base_lr, head_lr=head_lr, attr_head_lr=attr_lr
            )

            # Collect ids of params already tracked by the optimizer
            tracked = {id(p) for g in optimizer.param_groups for p in g['params']}

            # Add ONLY the new params (e.g., lora_*_t{task_id}) that aren't tracked yet
            added = 0
            for g in fresh_groups:
                new_params = [p for p in g['params'] if id(p) not in tracked]
                if not new_params:
                    continue
                ng = {k: v for k, v in g.items() if k != 'params'}
                ng['params'] = new_params
                optimizer.add_param_group(ng)
                added += len(new_params)

            if added > 0:
                print(f"[Task {task_id}] Added {added} new LoRA/head params to optimizer (no reinit).")

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            # Calculate learning rates based on multipliers  
            base_lr = args.lr
            attr_lr = base_lr * getattr(args, 'attr_lr_multiplier', 1.0)
            head_lr = base_lr * getattr(args, 'head_lr_multiplier', 1.0)
            
            # Get parameter groups with different learning rates (use non-DDP model)
            param_groups = model_without_ddp.get_parameter_groups(
                lora_lr=base_lr,
                head_lr=head_lr, 
                attr_head_lr=attr_lr
            )
            
            # Create new optimizer with parameter groups
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=getattr(args, 'weight_decay', 0.01)
            )

            # Rebuild scheduler to point to the NEW optimizer
            if lr_scheduler is not None:
                try: 
                    del lr_scheduler
                except: 
                    pass
                lr_scheduler = None
            
            # Recreate scheduler if needed
            from timm.scheduler import create_scheduler
            if args.sched != "constant":
                lr_scheduler, _ = create_scheduler(args, optimizer)

            optimizer.zero_grad(set_to_none=True)
            print(f"[Task {task_id}] Reinitialized optimizer and scheduler.")

        # Initialize for new task
        dataset_str = {"czsl-clothing16k":"clothing16k",
                       "czsl-ut-zappos":"ut-zappos",
                       "czsl-cgqa":"cgqa"}[args.dataset]

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=data_loader[task_id]["train"],
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_grad,
                set_training_mode=True,
                task_id=task_id,
                class_mask=class_mask,
                args=args,
            )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        # Build/Update prototypes if using prototype head
        if hasattr(model, 'use_proto_head') and model.use_proto_head:
            print(f"\n=== Building Full Prototype Matrix (Task {task_id + 1}) ===")
            
            # Get current task class indices
            idx = torch.as_tensor(class_mask[task_id], device=device, dtype=torch.long)
            
            # =====================================================================
            # STEP 1: Build DIAGONAL - P[task_id, task_id] (native prototypes)
            # =====================================================================
            print(f"\n[1/4] Computing diagonal: P[{task_id}, {task_id}] (native)")
            
            protos_native = build_prototypes_from_loaders(
                model=model,
                loaders=[data_loader[task_id]],
                device=device,
                num_classes=args.nb_classes,
                split=getattr(args, 'proto_from', 'seen_train'),
                task_ids=[task_id]  # Native adapter
            )
            
            # Apply PGWS blending to native prototypes
            if getattr(args, 'pgws_proto_blend', True):
                dataset_str = {"czsl-clothing16k":"clothing16k",
                               "czsl-ut-zappos":"ut-zappos",
                               "czsl-cgqa":"cgqa"}[args.dataset]
                
                eta_both = getattr(args, 'pgws_eta_both', 0.20)
                eta_one  = getattr(args, 'pgws_eta_one',  0.10)
                eta_none = getattr(args, 'pgws_eta_none', 0.00)
                
                protos_native = pgws_blend_into_prototypes(
                    protos_native, model_without_ddp, class_mask, task_id, dataset_str,
                    eta_both=eta_both, eta_one=eta_one, eta_none=eta_none
                )
            
            # Store diagonal
            model_without_ddp.store_prototypes_in_matrix(
                task_id, task_id, protos_native, idx
            )
            
            # Keep old prototypes buffer updated for backward compatibility
            if task_id == 0:
                model_without_ddp.prototypes.copy_(protos_native)
                model_without_ddp.prototypes_ready.fill_(True)
            
            # Snapshot for future AST transport
            model_without_ddp.snapshot_task_prototypes(task_id, protos_native, idx)
            
            # =====================================================================
            # STEP 2: Build BELOW DIAGONAL - P[task_id, 0...task_id-1]
            # =====================================================================
            print(f"\n[2/4] Computing below diagonal: P[{task_id}, 0..{task_id-1}]")
            
            for old_space in range(task_id):
                print(f"  Computing P[{task_id}, {old_space}] (current images → old adapter {old_space})")
                
                protos_cur_in_old = build_prototypes_from_loaders(
                    model=model,
                    loaders=[data_loader[task_id]],  # Current task images
                    device=device,
                    num_classes=args.nb_classes,
                    split=getattr(args, 'proto_from', 'seen_train'),
                    task_ids=[old_space]  # OLD adapter
                )
                
                # Store below diagonal entry
                model_without_ddp.store_prototypes_in_matrix(
                    task_id, old_space, protos_cur_in_old, idx
                )
            
            # =====================================================================
            # STEP 3: Transport ABOVE DIAGONAL - P[0...task_id-1, task_id]
            # =====================================================================
            print(f"\n[3/4] Transporting above diagonal: P[0..{task_id-1}, {task_id}] (AST)")
            
            if task_id > 0:
                for old_task in range(task_id):
                    print(f"  Transporting P[{old_task}, {old_task}] → P[{old_task}, {task_id}]")
                    
                    # Learn affine transport: old_space → current_space
                    A, b = _fit_affine_transport(
                        model=model_without_ddp,
                        loader_cur_task=data_loader[task_id],
                        src_task_id=old_task,
                        tgt_task_id=task_id,
                        device=device,
                        max_pairs=getattr(args, 'transport_max_pairs', 4096),
                        ridge=getattr(args, 'transport_ridge', 1e-3)
                    )
                    
                    # Get old prototypes from snapshot (native space)
                    old_snapshot = model_without_ddp.proto_snapshots[old_task]
                    old_protos_native = old_snapshot['rows']  # [K_old, D]
                    old_class_ids = old_snapshot['class_ids']  # [K_old]
                    
                    # Transport to current space
                    transported_protos = (old_protos_native @ A.T) + b
                    transported_protos = transported_protos / (
                        transported_protos.norm(dim=1, keepdim=True) + 1e-8
                    )
                    
                    # Create full matrix to store
                    full_matrix = torch.zeros(args.nb_classes, model_without_ddp.embed_dim, 
                                             device=device, dtype=transported_protos.dtype)
                    full_matrix[old_class_ids] = transported_protos
                    
                    # Store above diagonal entry
                    model_without_ddp.store_prototypes_in_matrix(
                        old_task, task_id, full_matrix, old_class_ids
                    )
                    
            
            # =====================================================================
            # STEP 4: Summary
            # =====================================================================
            print(f"\n[4/4] Prototype matrix summary:")
            print(f"  Total entries: {len(model_without_ddp.proto_matrix)}")
            print(f"  Matrix structure after task {task_id}:")
            for t in range(task_id + 1):
                row_str = f"    Task {t}: "
                for s in range(task_id + 1):
                    if (t, s) in model_without_ddp.proto_matrix:
                        shape = model_without_ddp.proto_matrix[(t, s)].shape
                        row_str += f"[{shape[0]}x{shape[1]}] "
                    else:
                        row_str += "[ --- ] "
                print(row_str)
            
            print("="*70)


        test_stats = evaluate_till_now(
            model=model,
            data_loader=data_loader,
            device=device,
            task_id=task_id,
            class_mask=class_mask,
            acc_matrix=acc_matrix,
            attr_acc_matrix=attr_acc_matrix,
            obj_acc_matrix=obj_acc_matrix,
            args=args,
        )

        

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, "checkpoint")).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, "checkpoint/task{}_checkpoint.pth".format(task_id + 1))
            state_dict = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if args.sched is not None and args.sched != "constant":
                state_dict["lr_scheduler"] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and utils.is_main_process():
            with open(
                os.path.join(
                    args.output_dir, "{}_stats.txt".format(datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M"))
                ),
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")


@torch.no_grad()
def pgws_blend_into_prototypes(
    protos: torch.Tensor,
    model,                # model_without_ddp
    class_mask,
    task_id: int,
    dataset: str,
    eta_both: float = 0.20,
    eta_one: float  = 0.10,
    eta_none: float = 0.00,   # if neither primitive seen before; keep 0 unless want to nudge
):
    """
    For each new composition class of this task, blend a PGWS direction
    (unit(Wa[a]) + unit(Wo[o])) into the empirical prototype before set_prototypes().

    - If BOTH primitives were seen in tasks < task_id: use eta_both
    - If EXACTLY ONE primitive was seen:               use eta_one
    - If NONE were seen:                                use eta_none (default 0: no blend)
    """
    if class_mask is None:
        return protos

    mapping = get_mapping(dataset)   # comp_id -> [attr_id, obj_id]
    Wa = model.attribute_head.weight
    Wo = model.object_head.weight

    # Unit rows to avoid head-scale dominance
    Wa_dir = Wa / (Wa.norm(dim=1, keepdim=True) + 1e-8)
    Wo_dir = Wo / (Wo.norm(dim=1, keepdim=True) + 1e-8)

    # --- Figure out which primitives have been seen BEFORE this task ---
    seen_attrs, seen_objs = set(), set()
    for t in range(task_id):                  # only past tasks
        for c_prev in class_mask[t]:
            a_prev, o_prev = mapping[int(c_prev)]
            seen_attrs.add(int(a_prev))
            seen_objs.add(int(o_prev))

    comp_ids = class_mask[task_id]
    blended_classes = []
    cosine_sims = []
    count_both = count_one = count_none = 0

    for c in comp_ids:
        a, o = mapping[int(c)]
        both = (a in seen_attrs) and (o in seen_objs)
        one  = ((a in seen_attrs) ^ (o in seen_objs))  # xor
        if both:
            eta = eta_both; count_both += 1
        elif one:
            eta = eta_one;  count_one  += 1
        else:
            eta = eta_none; count_none += 1

        if eta <= 0.0:
            # keep prototype as-is
            continue

        # PGWS direction
        d = Wa_dir[a] + Wo_dir[o]
        d = d / (d.norm() + 1e-8)

        p_emp = protos[int(c)]
        p_emp_norm = p_emp / (p_emp.norm() + 1e-8)

        blended = (1.0 - eta) * p_emp_norm + eta * d
        protos[int(c)] = blended / (blended.norm() + 1e-8)

        cosine_sims.append(F.cosine_similarity(
            protos[int(c)].unsqueeze(0), d.unsqueeze(0)).item())
        blended_classes.append(int(c))

    if blended_classes:
        avg_cos = sum(cosine_sims) / len(cosine_sims)
        print(
            f"[PGWS Proto] Task {task_id+1}: blended {len(blended_classes)} protos "
            f"(both/one/none = {count_both}/{count_one}/{count_none}; "
            f"eta_both={eta_both:.2f}, eta_one={eta_one:.2f}, eta_none={eta_none:.2f}); "
            f"avg cosine to PGWS dir: {avg_cos:.4f}"
        )
    else:
        print(f"[PGWS Proto] Task {task_id+1}: no blending applied")

    return protos



def get_mapping(dataset):
    fpath = "local_datasets/" + dataset + "/com2pri.txt"
    with open(fpath, "r") as f:
        mapping_content = f.read()

    mapping_dict = {}
    lines = mapping_content.split("\n")
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            key = int(parts[0])
            values = list(map(int, parts[1].split(",")))
            mapping_dict[key] = values
    return mapping_dict


def get_primitives_gt(dataset, mb_y):
    if dataset.startswith("czsl-"):
        dataset = dataset.lstrip("czsl-")
    mapping_dict = get_mapping(dataset)
    attr_gt = torch.zeros_like(mb_y)
    obj_gt = torch.zeros_like(mb_y)

    for i, val in enumerate(mb_y):
        mapping = mapping_dict[val.item()]
        attr_gt[i] = mapping[0]
        obj_gt[i] = mapping[1]

    return attr_gt, obj_gt




def get_logits_primitives(dataset, mb_y):
    mapping_dict = get_mapping(dataset)
    attribute_gt = torch.zeros_like(mb_y)
    object_gt = torch.zeros_like(mb_y)

    for i, val in enumerate(mb_y.view(-1)):
        mapping = mapping_dict[val.item()]
        attribute_gt.view(-1)[i] = mapping[0]
        object_gt.view(-1)[i] = mapping[1]

    attribute_gt = attribute_gt.view(mb_y.size())
    object_gt = object_gt.view(mb_y.size())

    return attribute_gt, object_gt


def accuracy_primitives(output, target, datasets, primitives, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    attribute_pre, object_pre = get_logits_primitives(datasets, pred)
    if primitives == "attr":
        pred = attribute_pre.t()
    else:
        pred = object_pre.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]

def compute_harmonic_mean(attr_acc, obj_acc):
    if attr_acc + obj_acc == 0:
        return 0.0
    return 2 * attr_acc * obj_acc / (attr_acc + obj_acc)
