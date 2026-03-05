from timm.models.registry import register_model

from vision_transformer_lora import _create_vision_transformer

__all__ = [
    "vit_tiny_patch16_224_lora",
    "vit_small_patch16_224_lora", 
    "vit_base_patch16_224_lora",
]


@register_model
def vit_tiny_patch16_224_lora(pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16) with LoRA adaptation"""
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer("vit_tiny_patch16_224", pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_lora(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16) with LoRA adaptation"""
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer("vit_small_patch16_224", pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_lora(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) with LoRA adaptation"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
    return model