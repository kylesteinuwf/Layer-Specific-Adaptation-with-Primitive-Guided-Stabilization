import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import (
    build_model_with_cfg,
    resolve_pretrained_cfg,
    named_apply,
    adapt_input_conv,
    checkpoint_seq,
)
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

# Import LoRA instead of Prompt
from lora import CompILerLoRA, apply_lora_to_vit, freeze_non_lora_parameters, print_lora_info

_logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    "vit_tiny_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    # 'vit_base_patch16_224': _cfg(
    #     url='https://storage.googleapis.com/vit_models/augreg/'
    #         'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    "vit_base_patch16_224": _cfg(url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"),
    
}

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        init_values=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformerLoRA(nn.Module):
    """Vision Transformer with LoRA adaptation (replaces CompILer's prompt system)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        # Prompt parameters (ignored for LoRA compatibility)
        prompt_length=None,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        head_type="token",
        use_prompt_mask=False,
        # LoRA-specific parameters
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        use_lora=True,
    ):
        """
        Args:
            All standard ViT parameters plus:
            lora_rank (int): LoRA rank
            lora_alpha (float): LoRA alpha parameter
            lora_dropout (float): LoRA dropout rate
            use_lora (bool): Enable LoRA adaptation
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.class_token = class_token
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        
        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        
        # LoRA doesn't add prompt tokens, so embed_len stays standard
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Store head type for compatibility
        self.head_type = head_type

        # Create LoRA object
        if use_lora:
            self.lora = CompILerLoRA(
                length=5,  # Not used in LoRA
                embed_dim=embed_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        # Dataset-specific heads (matches CompILer)
        if num_classes == 35:  # clothing16k
            self.head = nn.Linear(self.embed_dim, num_classes)
            self.attr_head = nn.Linear(self.embed_dim, 9)
            self.obj_head = nn.Linear(self.embed_dim, 8)
            self.dataset = "clothing16k"
        elif num_classes == 80:  # ut-zappos
            self.head = nn.Linear(self.embed_dim, num_classes)
            self.attr_head = nn.Linear(self.embed_dim, 15)
            self.obj_head = nn.Linear(self.embed_dim, 12)
            self.dataset = "ut-zappos"
        else:
            self.head = nn.Identity()
            self.attr_head = nn.Identity()
            self.obj_head = nn.Identity()

        # Apply LoRA to attention layers (will be done later in _create_vision_transformer if needed)
        if use_lora:
            # LoRA will be applied after pretrained weights are loaded
            pass

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if mode.startswith("jax"):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
            if self.cls_token is not None:
                nn.init.normal_(self.cls_token, std=1e-6)
            named_apply(partial(_init_vit_weights, head_bias=head_bias), self)

    def _init_weights(self, m):
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)

        # Initialize result dict
        res = dict()
        
        # LoRA doesn't add tokens, so these are always 0
        self.total_prompt_len = 0
        self.attr_total_prompt_len = 0
        self.obj_total_prompt_len = 0

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)  # LoRA is applied here automatically

        x = self.norm(x)
        res["x"] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res["x"]
        
        # For LoRA, we use standard CLS token approach regardless of head_type
        if self.class_token:
            # Use CLS token for all predictions
            cls_token = x[:, 0]  # CLS token is always first
            res["pre_logits"] = cls_token
            
            # Apply normalization
            cls_token = self.fc_norm(cls_token)
            
            # Generate all three logits from CLS token
            res["com_logits"] = self.head(cls_token)
            res["attr_logits"] = self.attr_head(cls_token)
            res["obj_logits"] = self.obj_head(cls_token)
            
        elif self.global_pool == "avg":
            # Use global average pooling if no CLS token
            x_pooled = x.mean(dim=1)
            res["pre_logits"] = x_pooled
            
            x_pooled = self.fc_norm(x_pooled)
            res["com_logits"] = self.head(x_pooled)
            res["attr_logits"] = self.attr_head(x_pooled)
            res["obj_logits"] = self.obj_head(x_pooled)
        else:
            raise ValueError("Either class_token must be True or global_pool must be 'avg'")

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res


def _init_vit_weights(module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False):
    """ViT weight initialization"""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=(), gs_old=()):
    """Resizes position embeddings"""
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = (int(math.sqrt(len(posemb_grid))),) * 2 if not gs_old else gs_old
    gs_new = (int(math.sqrt(ntok_new)),) * 2 if not gs_new else gs_new
    posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode="bicubic", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                getattr(model, "num_prefix_tokens", 1),
                model.patch_embed.grid_size,
                getattr(model, "patch_size", 16),
            )
        out_dict[k] = v

    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None))
    
    # Load pretrained weights before applying LoRA
    temp_kwargs = kwargs.copy()
    temp_kwargs['use_lora'] = False  # Don't apply LoRA yet
    
    model = build_model_with_cfg(
        VisionTransformerLoRA,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg.get("url", ""),
        **temp_kwargs,
    )
    
    # Now apply LoRA after pretrained weights are loaded
    if kwargs.get('use_lora', True):
        from lora import apply_lora_to_vit, freeze_non_lora_parameters, print_lora_info
        model.use_lora = True
        model.lora_layers = apply_lora_to_vit(
            model, 
            rank=kwargs.get('lora_rank', 8), 
            alpha=kwargs.get('lora_alpha', 16.0), 
            dropout=kwargs.get('lora_dropout', 0.0)
        )
        freeze_non_lora_parameters(model)
        print_lora_info(model)
    
    return model


# Model variants
@register_model
def vit_base_patch16_224_lora(pretrained=False, **kwargs):
    """ViT-Base model with LoRA adaptation."""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
    return model