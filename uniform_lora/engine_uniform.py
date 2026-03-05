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
from timm.optim import create_optimizer

import utils
from lora_uniform import (get_primitives_in_this_experience, get_primitives_gt, 
                  compute_final_score_train, compute_final_score, 
                  build_prototypes_from_loaders, apply_compiler_masking_train,
                  apply_compiler_masking_eval)

softmax = nn.Softmax(dim=1)


def compute_harmonic_mean(attr_acc, obj_acc):
    """
    Compute harmonic mean of attribute and object accuracies.
    Harmonic mean is useful for compositional learning as it emphasizes balanced performance.
    
    Args:
        attr_acc: Attribute accuracy (float)
        obj_acc: Object accuracy (float)
    
    Returns:
        Harmonic mean of the two accuracies (float)
    """
    if attr_acc + obj_acc == 0:
        return 0.0
    return 2 * attr_acc * obj_acc / (attr_acc + obj_acc)


def train_one_epoch(
    model: torch.nn.Module,
    original_model: torch.nn.Module,
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
    original_model.eval()

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
    
    # Use standard CrossEntropy loss for all three heads
    CE_com = nn.CrossEntropyLoss()
    CE_attr = nn.CrossEntropyLoss()
    CE_obj = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("Loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    
    header = f"Train: [Task {task_id+1}] Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]"

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output["pre_logits"]
            else:
                cls_features = None

        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)

        logits = output["com_logits"]
        attr_logits = output["attr_logits"]
        obj_logits = output["obj_logits"]

        # Apply CompILer's exact training masking logic using dedicated function
        logits, attr_logits, obj_logits = apply_compiler_masking_train(
            logits, attr_logits, obj_logits, dataset, task_id, class_mask, 
            attr_num, obj_num, args.nb_classes, args
        )

        # attr, obj gt and embddings
        attribute_gt, object_gt = get_primitives_gt(dataset, target)
        com_loss = CE_com(logits, target)
        attr_loss = CE_attr(attr_logits, attribute_gt)
        obj_loss = CE_obj(obj_logits, object_gt)
        
        # Simple loss combination like tri-lora version
        loss = getattr(args, 'lambda_comp', 0.5) * com_loss \
             + getattr(args, 'lambda_attr', 0.25) * attr_loss \
             + getattr(args, 'lambda_obj', 0.25) * obj_loss

        total_logits = compute_final_score_train(dataset, logits, attr_logits, obj_logits, args.beta)
        acc1, acc5 = accuracy(total_logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            message = "Loss is {}, stopping training".format(loss.item())
            print(message)
            wandb.alert(message)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
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
                f"Task-{task_id}/Com_Loss": com_loss.item(),
                f"Task-{task_id}/Attr_Loss": attr_loss.item(),
                f"Task-{task_id}/Obj_Loss": obj_loss.item(),
            }
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    data_loader,
    device,
    task_id=-1,
    class_mask=None,
    args=None,
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
    header = "Test: [Task {}]".format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            if original_model is not None:
                output = original_model(input)
                cls_features = output["pre_logits"]
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output["com_logits"]
            attr_logits = output["attr_logits"]
            obj_logits = output["obj_logits"]

            attribute_gt, object_gt = get_primitives_gt(dataset, target)

            # Apply CompILer's exact evaluation masking logic using dedicated function
            logits, attr_logits, obj_logits = apply_compiler_masking_eval(
                logits, attr_logits, obj_logits, dataset, task_id, class_mask,
                attr_num, obj_num, args.nb_classes, args
            )

            com_loss = CE_com(logits, target)
            attr_loss = CE_attr(attr_logits, attribute_gt)
            obj_loss = CE_obj(obj_logits, object_gt)
            loss = getattr(args, 'lambda_comp', 0.5) * com_loss \
                 + getattr(args, 'lambda_attr', 0.25) * attr_loss \
                 + getattr(args, 'lambda_obj', 0.25) * obj_loss

            total_logits = compute_final_score(dataset, logits, attr_logits, obj_logits, args.beta)
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
    original_model: torch.nn.Module,
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
            original_model=original_model,
            data_loader=data_loader[i]["val"],
            device=device,
            task_id=i,
            class_mask=class_mask,
            args=args,
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
    original_model: torch.nn.Module,
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

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)

        # Primitive-Guided Warm Start for new compositions in this task
        if getattr(args, 'pgws', True):  # default ON unless you add a flag
            dataset_str = {"czsl-clothing16k":"clothing16k",
                           "czsl-ut-zappos":"ut-zappos",
                           "czsl-cgqa":"cgqa"}[args.dataset]
            pgws_seed_new_comp_rows(model_without_ddp, class_mask, task_id, dataset_str)

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model=model,
                original_model=original_model,
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

        # Build prototypes if using prototype head
        if hasattr(model, 'use_proto_head') and model.use_proto_head:
            print(f"\nBuilding composite prototypes from seen data (task {task_id + 1})...")
            protos = build_prototypes_from_loaders(
                model=model,
                loaders=data_loader[:task_id+1],  # Only seen tasks
                device=device,
                num_classes=args.nb_classes,
                split=getattr(args, 'proto_from', 'seen_train')
            )

            # Blend PGWS direction into current task's prototypes
            if getattr(args, 'pgws_proto_blend', True):
                dataset_str = {"czsl-clothing16k":"clothing16k",
                            "czsl-ut-zappos":"ut-zappos",
                            "czsl-cgqa":"cgqa"}[args.dataset]
                eta_both = getattr(args, 'pgws_eta_both', 0.20)  
                eta_one  = getattr(args, 'pgws_eta_one',  0.10)
                eta_none = getattr(args, 'pgws_eta_none', 0.00)
                protos = pgws_blend_into_prototypes(
                    protos, model_without_ddp, class_mask, task_id, dataset_str,
                    eta_both=eta_both, eta_one=eta_one, eta_none=eta_none
                )

            model.set_prototypes(protos)
            print("✓ Prototypes updated")

        test_stats = evaluate_till_now(
            model=model,
            original_model=original_model,
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


def get_primitives_in_this_experience(dataset, mask):
    mapping_dict = get_mapping(dataset)
    attr_in_this_experience = [mapping_dict[key][0] for key in mask]
    obj_in_this_experience = [mapping_dict[key][1] for key in mask]
    attr_in_this_experience = list(set(attr_in_this_experience))
    obj_in_this_experience = list(set(obj_in_this_experience))

    return attr_in_this_experience, obj_in_this_experience


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


def compute_final_score_train(dataset, com_logits, attr_logits, obj_logits, beta):
    mapping_dict = get_mapping(dataset)
    dev = com_logits.device
    attr = torch.tensor([v[0] for v in mapping_dict.values()], device=dev, dtype=torch.long)
    obj  = torch.tensor([v[1] for v in mapping_dict.values()], device=dev, dtype=torch.long)

    attr_logit = attr_logits[:, attr]
    obj_logit  = obj_logits[:, obj]

    fused = com_logits + beta * (attr_logit + obj_logit)

    # keep masked -inf entries as-is
    mask = torch.isfinite(com_logits)
    return torch.where(mask, fused, com_logits)


def compute_final_score(dataset, com_logits, attr_logits, obj_logits, beta):
    mapping_dict = get_mapping(dataset)
    dev = com_logits.device
    attr = torch.tensor([v[0] for v in mapping_dict.values()], device=dev, dtype=torch.long)
    obj  = torch.tensor([v[1] for v in mapping_dict.values()], device=dev, dtype=torch.long)

    attr_logit = attr_logits[:, attr]
    obj_logit  = obj_logits[:, obj]
    return com_logits + beta * (attr_logit + obj_logit)


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


@torch.no_grad()
def pgws_seed_new_comp_rows(model, class_mask, task_id, dataset):
    """
    Primitive-Guided Warm Start (PGWS):
    For each composition class introduced in this task, initialize its
    composition_head row to the normalized sum of the corresponding
    attribute/object head rows.
    """
    if class_mask is None:
        return
    comp_ids = class_mask[task_id]           # composition class IDs for current task
    mapping  = get_mapping(dataset)          # comp_id -> [attr_id, obj_id]
    Wc = model.composition_head.weight       # [C, D]
    Wa = model.attribute_head.weight         # [A, D]
    Wo = model.object_head.weight            # [O, D]

    # Unit-direction sum for stability across head scales
    Wa_dir = Wa / (Wa.norm(dim=1, keepdim=True) + 1e-8)
    Wo_dir = Wo / (Wo.norm(dim=1, keepdim=True) + 1e-8)

    seeded_classes = []
    cosine_similarities = []
    
    for c in comp_ids:
        a, o = mapping[int(c)]
        direction = Wa_dir[a] + Wo_dir[o]
        direction = direction / (direction.norm() + 1e-8)
        
        # Compute cosine similarity for verification
        original_sum = Wa_dir[a] + Wo_dir[o]
        original_sum_norm = original_sum / (original_sum.norm() + 1e-8)
        cosine_sim = F.cosine_similarity(direction.unsqueeze(0), original_sum_norm.unsqueeze(0)).item()
        
        Wc[int(c)].copy_(direction)
        seeded_classes.append(int(c))
        cosine_similarities.append(cosine_sim)
    
    if seeded_classes:
        avg_cosine = sum(cosine_similarities) / len(cosine_similarities)
        print(f"[PGWS] Task {task_id+1}: Seeded {len(seeded_classes)} compositions "
              f"(e.g., classes {seeded_classes[:3]}...), avg cosine: {avg_cosine:.4f}")


@torch.no_grad()
def pgws_blend_into_prototypes(
    protos: torch.Tensor,
    model,                # model_without_ddp
    class_mask,
    task_id: int,
    dataset: str,
    eta_both: float = 0.20,
    eta_one: float  = 0.10,
    eta_none: float = 0.00,   # if neither primitive seen before; keep 0 unless you want to nudge
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