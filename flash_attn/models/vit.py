# Copyright (c) 2022, Tri Dao.
# Inspired by / adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
import math
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from einops import rearrange

from timm.models.helpers import named_apply
from flash_attn.layers.patch_embed import PatchEmbed

from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp, FusedDenseGeluDense
from flash_attn.modules.block import Block

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None


def create_mixer_cls(num_heads, qkv_bias, attn_drop, use_flash_attn, fused_bias_fc,
                     cross_attn=False):
    mixer_cls = partial(MHA, num_heads=num_heads, cross_attn=cross_attn, bias=qkv_bias,
                        dropout=attn_drop, fused_bias_fc=fused_bias_fc,
                        use_flash_attn=use_flash_attn)
    return mixer_cls


def create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_dense_gelu_dense):
    inner_dim = int(embed_dim * mlp_ratio)
    if not fused_dense_gelu_dense:
        mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=act_layer())
    else:
        mlp_cls = partial(FusedDenseGeluDense, hidden_features=inner_dim)
    return mlp_cls


def create_block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path,
                 norm_layer, act_layer, use_flash_attn, fused_bias_fc, fused_dense_gelu_dense,
                 fused_dropout_add_ln, layer_idx=None, n_layer=None, last_layer_subset=False):
    mixer_cls = create_mixer_cls(num_heads, qkv_bias, attn_drop_rate, use_flash_attn, fused_bias_fc,
                                 cross_attn=(last_layer_subset and layer_idx == n_layer - 1))
    mlp_cls = create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_dense_gelu_dense)
    block = Block(embed_dim, mixer_cls, mlp_cls, norm_cls=norm_layer,
                  prenorm=True, resid_dropout=drop_rate, drop_path=drop_path,
                  fused_dropout_add_ln=fused_dropout_add_ln)
    return block


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            use_flash_attn=False,
            fused_bias_fc=False,
            fused_dense_gelu_dense=False,
            fused_dropout_add_ln=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool == 'token', 'Only support pooling with CLS token'
        assert class_token
        assert init_values is None, 'LayerScale is not supported yet'
        assert weight_init == ''
        assert fc_norm is None
        # pre_norm seems redundant, as there's a LayerNorm right at the start of each block, idk
        assert not pre_norm
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class

        patch_embed_extra_kwargs = ({'fused_bias_fc': fused_bias_fc} if embed_layer is PatchEmbed
                                    else {})
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            **patch_embed_extra_kwargs
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Attn / MLP -> Dropout -> Add -> LN, returning both the residual branch (output of Add) and
        # the main branch (output of LN). The model definition is unchanged, but the mapping of the
        # nn.LayerNorm weights are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        # self.norm_0 is the first layer norm in the model, while self.norm
        # (in the pretrained weight) is the final layer norm.
        self.norm_0 = norm_layer(embed_dim)

        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError('dropout_add_layer_norm is not installed')

        self.blocks = nn.ModuleList([create_block(
            embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path=dpr[i],
            norm_layer=norm_layer, act_layer=act_layer, use_flash_attn=use_flash_attn,
            fused_bias_fc=fused_bias_fc, fused_dense_gelu_dense=fused_dense_gelu_dense,
            fused_dropout_add_ln=fused_dropout_add_ln, layer_idx=i, n_layer=depth,
            last_layer_subset=(global_pool == 'token')
        ) for i in range(depth)])

        # Classifier Head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode == ''
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return x

    def forward_features(self, x, all_tokens=True):
        """
        If all_tokens==False and self.global_pool == 'token', we only return the features for the
        cls token.
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        # TD [2022-10-15]: Force residual in fp32 in case of DeepSpeed
        if not self.fused_dropout_add_ln:
            residual = self.pos_drop(x).float()
            hidden_states = self.norm_0(residual.to(dtype=self.norm_0.weight.dtype))
        else:
            hidden_states, residual = dropout_add_layer_norm(
                x, None, self.norm_0.weight, self.norm_0.bias,
                self.pos_drop.p if self.training else 0.0, self.norm_0.eps, prenorm=True,
                residual_in_fp32=True
            )
        hidden_states = self.norm_0(residual.to(dtype=self.norm_0.weight.dtype))
        if self.global_pool != 'token' or all_tokens:
            for block in self.blocks:
                hidden_states, residual = block(hidden_states, residual)
        else:
            for block in self.blocks[:-1]:
                hidden_states, residual = block(hidden_states, residual)
            # For the last layer, we only want the 1st token of the output. So we do cross-attention
            # where the query is the 1st token and the key/value is the whole sequence.
            hidden_states_1st = rearrange(hidden_states[:, 0], 'b d -> b 1 d')
            residual_1st = rearrange(residual[:, 0], 'b d -> b 1 d')
            hidden_states, _ = self.blocks[-1](hidden_states_1st, residual_1st,
                                               mixer_kwargs={'x_kv': hidden_states})
        return hidden_states

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x, all_tokens=False)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model
