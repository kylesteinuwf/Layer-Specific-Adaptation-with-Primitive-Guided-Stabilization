import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager



class TriLoRALayer(nn.Module):
    """Three LoRA branches (attr/obj/comp) with per-task parameters."""
    def __init__(self, original_layer: nn.Linear,
                 enable_attr: bool, enable_obj: bool, enable_comp: bool,
                 rank_attr=8, alpha_attr=16.0,
                 rank_obj=8,  alpha_obj=16.0,
                 rank_comp=8, alpha_comp=16.0):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

        if enable_attr:   self.branch_type = 'attr'
        elif enable_obj:  self.branch_type = 'obj'
        elif enable_comp: self.branch_type = 'comp'
        else:             self.branch_type = None

        self.ranks  = {'attr': rank_attr, 'obj': rank_obj, 'comp': rank_comp}
        self.alphas = {'attr': alpha_attr, 'obj': alpha_obj, 'comp': alpha_comp}

        self.task_params = {'attr': {}, 'obj': {}, 'comp': {}}
        self.active_task = None  # set by model/set_active_task

    def add_task(self, task_id: int):
        if self.branch_type is None: return
        if task_id in self.task_params[self.branch_type]: return
        r = self.ranks[self.branch_type]
        if r <= 0: return
        a = self.alphas[self.branch_type]

        # create on same device/dtype as the wrapped Linear
        dev  = self.original_layer.weight.device
        dtyp = self.original_layer.weight.dtype
        A = nn.Parameter(torch.randn(r, self.original_layer.in_features, device=dev, dtype=dtyp) / r)
        B = nn.Parameter(0.01 * torch.randn(self.original_layer.out_features, r, device=dev, dtype=dtyp))
        scale = torch.tensor(a / r, device=dev, dtype=dtyp)

        # register with stable names so the optimizer can pick them up
        setattr(self, f"lora_A_{self.branch_type}_t{int(task_id)}", A)
        setattr(self, f"lora_B_{self.branch_type}_t{int(task_id)}", B)

        # freeze older tasks; train new one
        for (A_old, B_old, _) in self.task_params[self.branch_type].values():
            A_old.requires_grad_(False); B_old.requires_grad_(False)
        A.requires_grad_(True); B.requires_grad_(True)

        self.task_params[self.branch_type][int(task_id)] = (A, B, scale)



    def forward(self, x):
        out = self.original_layer(x)
        bt = self.branch_type
        if bt is None or self.active_task is None:
            return out
        params = self.task_params[bt].get(int(self.active_task), None)
        if params is None:
            return out
        A, B, scale = params  # tensors already on correct device
        return out + ((x @ A.T) @ B.T) * scale



def apply_tri_lora_to_vit(vit, depth_assign, ranks, alphas):
    """Attach Tri-LoRA to attention qkv for selected blocks."""
    layers = []
    for bi, block in enumerate(getattr(vit, 'blocks', [])):
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        enable = {
            'attr': bi in depth_assign['attr'],
            'obj':  bi in depth_assign['obj'],
            'comp': bi in depth_assign['comp'],
        }
        # Skip wrapping blocks that don't use any branch
        if not any(enable.values()):
            continue
        # Assert at most one branch per block (disjoint ranges)
        assert sum(enable.values()) <= 1, f"Block {bi} has overlapping branches: {enable}"
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            wrapper = TriLoRALayer(attn.qkv,
                                   enable_attr=enable['attr'], enable_obj=enable['obj'], enable_comp=enable['comp'],
                                   rank_attr=ranks['attr'], alpha_attr=alphas['attr'],
                                   rank_obj=ranks['obj'],   alpha_obj=alphas['obj'],
                                   rank_comp=ranks['comp'], alpha_comp=alphas['comp'])
            # make wrapper live where the original layer lives
            wrapper = wrapper.to(attn.qkv.weight.device)
            attn.qkv = wrapper
            layers.append(wrapper)
        else:
            # Fallback: split q/k/v linears
            for name in ('q', 'k', 'v'):
                if hasattr(attn, name) and isinstance(getattr(attn, name), nn.Linear):
                    orig_linear = getattr(attn, name)
                    wrapper = TriLoRALayer(orig_linear,
                                           enable_attr=enable['attr'], enable_obj=enable['obj'], enable_comp=enable['comp'],
                                           rank_attr=ranks['attr'], alpha_attr=alphas['attr'],
                                           rank_obj=ranks['obj'],   alpha_obj=alphas['obj'],
                                           rank_comp=ranks['comp'], alpha_comp=alphas['comp'])
                    # make wrapper live where the original layer lives
                    wrapper = wrapper.to(orig_linear.weight.device)
                    setattr(attn, name, wrapper)
                    layers.append(wrapper)
    return layers

# def orth_loss(self, cur_task_id: int, lambda_orth: float = 1e-4, branches=("comp",)):
#     if getattr(self, "branch_type", None) not in branches:
#         return self.original_layer.weight.new_zeros(())

#     cur = self.task_params[self.branch_type].get(int(cur_task_id), None)
#     if cur is None:
#         return self.original_layer.weight.new_zeros(())

#     A_t, B_t, _ = cur
#     prev_As = [A.detach() for tid,(A,B,_) in self.task_params[self.branch_type].items()
#                if int(tid) < int(cur_task_id)]
#     if not prev_As:
#         return A_t.new_zeros(())

#     A_prev = torch.cat(prev_As, dim=0)
#     with torch.no_grad():
#         Q_A = torch.linalg.qr(A_prev.T, mode='reduced').Q.T
#     loss = (A_t @ Q_A.T).pow(2).sum()


#     return lambda_orth * loss



def get_primitives_in_this_experience(dataset, mask):
    """Extract primitive masks - matches CompILer's exact function."""
    mapping_dict = get_mapping(dataset)
    attr_in_this_experience = [mapping_dict[key][0] for key in mask]
    obj_in_this_experience = [mapping_dict[key][1] for key in mask]
    attr_in_this_experience = list(set(attr_in_this_experience))
    obj_in_this_experience = list(set(obj_in_this_experience))

    return attr_in_this_experience, obj_in_this_experience


def get_mapping(dataset):
    """Load com2pri mapping - matches CompILer's exact function."""
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
    """Extract primitive ground truth - matches CompILer's exact function."""
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
    """Compute final score for training - matches CompILer's exact function."""
    mapping_dict = get_mapping(dataset)
    attr = torch.tensor([value[0] for value in mapping_dict.values()])
    obj = torch.tensor([value[1] for value in mapping_dict.values()])
    attr_logits_clone = attr_logits.clone()
    obj_logits_clone = obj_logits.clone()
    attr_logit = attr_logits_clone[:, attr]
    obj_logit = obj_logits_clone[:, obj]
    mask = com_logits != float("-inf")
    intermediate = torch.zeros_like(com_logits)
    intermediate[mask] = com_logits[mask] + beta * (attr_logit[mask] + obj_logit[mask])
    result = torch.where(mask, intermediate, com_logits)
    return result


def compute_final_score(dataset, com_logits, attr_logits, obj_logits, beta):
    """Compute final score for evaluation - matches CompILer's exact function."""
    mapping_dict = get_mapping(dataset)
    attr = torch.tensor([value[0] for value in mapping_dict.values()])
    obj = torch.tensor([value[1] for value in mapping_dict.values()])
    attr_logits_clone = attr_logits.clone()
    obj_logits_clone = obj_logits.clone()
    attr_logit = attr_logits_clone[:, attr]
    obj_logit = obj_logits_clone[:, obj]
    com_logit = com_logits + beta * (attr_logit + obj_logit)
    return com_logit



class CompILerTriLoRAViTModel(nn.Module):
    """
    ViT with three LoRA specializations (attr/obj/comp).
    Single forward pass with feature taps for soft specialization.
    """
    def __init__(self, vit_model, attr_num, obj_num, com_num,
                 ranks={'attr':8,'obj':8,'comp':8},
                 alphas={'attr':16.0,'obj':16.0,'comp':16.0},
                 assign='auto',  # or dict {'attr': set(int), 'obj': set(int), 'comp': set(int)}
                 use_proto_head=False, proto_tau=20.0):
        super().__init__()
        self.backbone = vit_model
        self.embed_dim = vit_model.embed_dim
        self.attr_num, self.obj_num, self.com_num = attr_num, obj_num, com_num
        self.use_proto_head = use_proto_head
        self.proto_tau = nn.Parameter(torch.tensor(float(proto_tau)), requires_grad=False)
        # Old single-space storage (keep for backward compatibility during transition)
        self.register_buffer("prototypes", torch.zeros(com_num, self.embed_dim))
        self.register_buffer("prototypes_ready", torch.tensor(False))

        # NEW: Full prototype matrix for EASE-style inference
        # proto_matrix[(task_i, space_j)] = prototypes for task_i measured in space_j
        self.proto_matrix = {}  # Dict: (int, int) -> Tensor[K, D]
        for p in self.backbone.parameters():
            p.requires_grad = False
        depth = len(getattr(self.backbone, 'blocks', []))
        if assign == 'auto':
            last_n_comp = 2 if depth >= 12 else max(1, depth // 6)
            attr_end = max(1, depth // 3)
            comp_idxs = set(range(depth - last_n_comp, depth))
            attr_idxs = set(range(0, attr_end))
            obj_idxs  = set(i for i in range(depth) if i not in comp_idxs and i >= attr_end)
            depth_assign = {'attr': attr_idxs, 'obj': obj_idxs, 'comp': comp_idxs}
        else:
            depth_assign = assign
        self.lora_layers = apply_tri_lora_to_vit(self.backbone, depth_assign, ranks=ranks, alphas=alphas)
        
        # Cache cut indices for feature taps (end of early/middle ranges)
        self.attr_last_idx = max(depth_assign['attr']) if depth_assign['attr'] else -1
        self.obj_last_idx  = max(depth_assign['obj'])  if depth_assign['obj']  else -1
        
        # Track active task
        self.active_task = None
        
        self.composition_head = nn.Linear(self.embed_dim, com_num)
        self.attribute_head   = nn.Linear(self.embed_dim, attr_num)
        self.object_head      = nn.Linear(self.embed_dim, obj_num)
        
    def register_new_task(self, task_id: int):
        for w in self.lora_layers:
            w.add_task(task_id)

    def set_active_task(self, task_id: int):
        for w in self.lora_layers:
            w.active_task = int(task_id)
        self.active_task = int(task_id)

    def forward(self, x, task_id=-1, cls_features=None, train=None):
        # Ensure forward respects explicit task_id even when called with task_id=None after context manager
        if task_id is not None and int(task_id) >= 0:
            self.set_active_task(int(task_id))
        
        # --- embed + pos (mirror timm ViT.forward_features up to the block loop) ---
        xb = self.backbone.patch_embed(x)
        if hasattr(self.backbone, 'cls_token'):
            cls = self.backbone.cls_token.expand(xb.shape[0], -1, -1)
            xb = torch.cat((cls, xb), dim=1)
        if hasattr(self.backbone, 'pos_embed'):
            xb = xb + self.backbone.pos_embed
        if hasattr(self.backbone, 'pos_drop'):
            xb = self.backbone.pos_drop(xb)

        # figure out cut points based on your assignment
        # (computed in __init__; see the __init__ change below)
        feats_attr = None
        feats_obj  = None

        for i, blk in enumerate(self.backbone.blocks):
            xb = blk(xb)
            if i == self.attr_last_idx:
                feats_attr = xb
            if i == self.obj_last_idx:
                feats_obj = xb

        # final norm (timm does norm at end)
        def maybe_norm(t):
            return self.backbone.norm(t) if hasattr(self.backbone, 'norm') else t

        feats_comp = maybe_norm(xb)
        if feats_attr is not None:
            feats_attr = maybe_norm(feats_attr)
        if feats_obj is not None:
            feats_obj = maybe_norm(feats_obj)

        # pooling identical to your _pool
        def pool(feats):
            if feats is None:
                return None
            if getattr(self.backbone, 'global_pool', None) == 'avg':
                return feats[:, 1:].mean(dim=1)
            return feats[:, 0]

        pre_a = pool(feats_attr)
        pre_o = pool(feats_obj)
        pre_c = pool(feats_comp)

        # heads
        attr_logits = self.attribute_head(pre_a) if pre_a is not None else None
        obj_logits  = self.object_head(pre_o)    if pre_o is not None else None

        is_train = self.training if train is None else bool(train)
        use_proto = (self.use_proto_head and bool(self.prototypes_ready.item()) and (not is_train))
        if use_proto:
            z = self._l2norm(pre_c)
            com_logits = self.proto_tau * (z @ self.prototypes.t())
        else:
            com_logits = self.composition_head(pre_c)

        return {
            'com_logits':  com_logits,
            'attr_logits': attr_logits,
            'obj_logits':  obj_logits,
            'pre_logits':  pre_c,  # for prototype building
        }
    def get_parameter_groups(self, lora_lr=1e-4, head_lr=5e-4, attr_head_lr=1e-3,
                             lora_attr_mult=1.0, lora_obj_mult=1.0, lora_comp_mult=1.0, task_id=None):
        groups, attr_params, obj_params, comp_params = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: 
                continue
            # Match per-task LoRA parameters
            if any(k in n for k in ['lora_A_attr_t','lora_B_attr_t']): attr_params.append(p)
            elif any(k in n for k in ['lora_A_obj_t','lora_B_obj_t']): obj_params.append(p)
            elif any(k in n for k in ['lora_A_comp_t','lora_B_comp_t']): comp_params.append(p)
        if attr_params: groups.append({'params': attr_params, 'lr': lora_lr * lora_attr_mult, 'name':'lora_attr'})
        if obj_params:  groups.append({'params': obj_params,  'lr': lora_lr * lora_obj_mult,  'name':'lora_obj'})
        if comp_params: groups.append({'params': comp_params, 'lr': lora_lr * lora_comp_mult, 'name':'lora_comp'})
        groups.append({'params': self.attribute_head.parameters(), 'lr': attr_head_lr, 'name':'attribute_head'})
        groups.append({'params': list(self.composition_head.parameters()) + list(self.object_head.parameters()),
                       'lr': head_lr, 'name':'other_heads'})
        return groups
    
    def verify_frozen_backbone(self):
        """Assert that only LoRA and head parameters are trainable."""
        violations = []
        for name, param in self.named_parameters():
            is_lora = any(k in name for k in ['lora_A_attr_t', 'lora_B_attr_t', 'lora_A_obj_t', 'lora_B_obj_t', 'lora_A_comp_t', 'lora_B_comp_t'])
            is_head = any(head in name for head in ['composition_head', 'attribute_head', 'object_head'])
            
            if param.requires_grad and not (is_lora or is_head):
                violations.append(name)
        
        if violations:
            raise AssertionError(f"Unexpected trainable parameters (should be LoRA+heads only): {violations}")
        return True
    
    def _l2norm(self, x, dim=-1, eps=1e-8):
        """L2 normalize tensor along specified dimension."""
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    @torch.no_grad()
    def set_prototypes(self, protos: torch.Tensor):
        """Set prototypes and mark them as ready.
        
        Args:
            protos: [C, D] tensor of prototypes in same device/dtype as model
        """
        self.prototypes.copy_(self._l2norm(protos))
        self.prototypes_ready.fill_(True)
        print(f"✓ Set {protos.shape[0]} L2-normalized prototypes")
    
    @torch.no_grad() 
    def reset_prototypes(self):
        """Reset prototypes to zero and mark as not ready."""
        self.prototypes.zero_()
        self.prototypes_ready.fill_(False)
        print("✓ Reset prototypes")
    
    @contextmanager
    def _using_task(self, task_id: int):
        """Context manager to temporarily set active task and restore afterwards."""
        prev = getattr(self, "active_task", None)
        if task_id is not None:
            self.set_active_task(task_id)
        try:
            yield
        finally:
            # restore previous active task to avoid side-effects
            if prev is not None:
                self.set_active_task(prev)

    def extract_features(self, x, task_id: int):
        """Return L2-normalized pre_logits using a specified adapter (task_id)."""
        with self._using_task(task_id):
            out = self.forward(x, task_id=None, train=False)  # task_id handled by context manager
        return self._l2norm(out["pre_logits"])

    @torch.no_grad()
    def snapshot_task_prototypes(self, task_id: int, protos_tensor: torch.Tensor, class_ids: torch.LongTensor):
        """
        Save per-task prototype rows (as measured in task_id's subspace) for later transport.
        """
        if not hasattr(self, 'proto_snapshots'):
            self.proto_snapshots = {}  # task_id -> {'rows':[K,D], 'class_ids':[K]}
        rows = protos_tensor.index_select(0, class_ids).detach().clone()
        rows = self._l2norm(rows)
        self.proto_snapshots[int(task_id)] = {'rows': rows, 'class_ids': class_ids.detach().clone()}

    @torch.no_grad()
    def store_prototypes_in_matrix(self, task_id: int, space_id: int, 
                                   protos_tensor: torch.Tensor, 
                                   class_ids: torch.LongTensor):
        """
        Store prototypes for task_id measured in space_id's adapter.
        
        Args:
            task_id: Which task these prototypes belong to
            space_id: Which adapter space they were measured in
            protos_tensor: [C, D] full prototype matrix
            class_ids: [K] indices of classes for this task
        """
        rows = protos_tensor.index_select(0, class_ids).detach().clone()
        rows = self._l2norm(rows)
        self.proto_matrix[(int(task_id), int(space_id))] = rows
        print(f"Stored proto_matrix[({task_id}, {space_id})]: {rows.shape}")


@torch.no_grad()
def build_prototypes_from_loaders(model, loaders, device, num_classes, split='seen_train', task_ids=None):
    """
    Aggregate L2-normalized pre_logits to class means over all tasks seen so far.
    
    Args:
        model: CompILerLoraViTModel instance
        loaders: list of dicts per task, each with keys 'train' and 'val'
        device: device to run on
        num_classes: total number of composition classes
        split: 'seen_train' -> use train splits; 'seen_val' -> val splits
        task_ids: list of actual task IDs corresponding to loaders (if None, uses range(len(loaders)))
        
    Returns:
        prototypes: [num_classes, embed_dim] tensor of class prototypes
    """
    model.eval()
    sums = torch.zeros(num_classes, model.embed_dim, device=device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    # Choose which key we aggregate from
    key = 'train' if split == 'seen_train' else 'val'

    for i in range(len(loaders)):
        dl = loaders[i][key]
        real_tid = task_ids[i] if task_ids is not None else i
        for inputs, targets in dl:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            out = model(inputs, task_id=real_tid)  # Use correct task-specific adapter
            z = out['pre_logits']
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)  # L2 normalize

            # Accumulate
            sums.index_add_(0, targets, z)
            counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.long))

    # Avoid div-by-zero (unseen classes stay zero)
    mask = counts > 0
    protos = torch.zeros_like(sums)
    protos[mask] = sums[mask] / counts[mask].unsqueeze(1)
    
    print(f"Built prototypes for {mask.sum().item()}/{num_classes} classes from {len(loaders)} tasks")
    return protos



def apply_compiler_masking_train(logits, attr_logits, obj_logits, dataset, task_id, class_mask, 
                                attr_num, obj_num, nb_classes, args):
    """Apply CompILer's exact training masking logic."""
    # here is the trick to mask out classes of non-current tasks
    if args.train_mask and class_mask is not None:
        mask = class_mask[task_id]
        attr_mask, obj_mask = get_primitives_in_this_experience(dataset, mask)
        
        device = logits.device
        
        # Composition masking
        not_mask = np.setdiff1d(np.arange(nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))
        
        # Attribute masking
        attr_not_mask = np.setdiff1d(np.arange(attr_num), attr_mask)
        attr_not_mask = torch.tensor(attr_not_mask, dtype=torch.int64).to(device)
        attr_logits = attr_logits.index_fill(dim=1, index=attr_not_mask, value=float("-inf"))
        
        # Object masking
        obj_not_mask = np.setdiff1d(np.arange(obj_num), obj_mask)
        obj_not_mask = torch.tensor(obj_not_mask, dtype=torch.int64).to(device)
        obj_logits = obj_logits.index_fill(dim=1, index=obj_not_mask, value=float("-inf"))
        
    return logits, attr_logits, obj_logits


def apply_compiler_masking_eval(logits, attr_logits, obj_logits, dataset, task_id, class_mask,
                               attr_num, obj_num, nb_classes, args):
    """Apply CompILer's exact evaluation masking logic."""

    # Only assert when masking is actually supposed to run
    if args.task_inc and class_mask is not None:
        assert args.task_inc is True, "Expected TI eval; pass --task_inc True"
        # adding mask to output logits
        mask = class_mask[task_id]
        attr_mask, obj_mask = get_primitives_in_this_experience(dataset, mask)
        device = logits.device

        # Composition masking
        not_mask = np.setdiff1d(np.arange(nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        # Attribute masking
        attr_not_mask = np.setdiff1d(np.arange(attr_num), attr_mask)
        attr_not_mask = torch.tensor(attr_not_mask, dtype=torch.int64).to(device)
        attr_logits = attr_logits.index_fill(dim=1, index=attr_not_mask, value=float("-inf"))

        # Object masking
        obj_not_mask = np.setdiff1d(np.arange(obj_num), obj_mask)
        obj_not_mask = torch.tensor(obj_not_mask, dtype=torch.int64).to(device)
        obj_logits = obj_logits.index_fill(dim=1, index=obj_not_mask, value=float("-inf"))

    return logits, attr_logits, obj_logits



def print_lora_info(model):
    """Print information about LoRA layers in the model."""
    lora_params = 0
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += param.numel()
    
    print(f"LoRA Parameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Parameter efficiency: {100 * trainable_params / total_params:.2f}%")