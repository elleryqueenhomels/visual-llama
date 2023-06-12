# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass, field
import math

import clip

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    adapter_len: int = 10
    adapter_layer: int = 30

    add_bias: bool = False
    add_scale: bool = False

    use_lora: bool = False
    lora_rank: int = 16

    vision_clip_model: str = "ViT-L/14"
    vision_dim: int = 512
    vision_blocks: int = 2
    vision_early_fusion: set = field(default_factory=set)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def forward_linear_with_scale_and_bias(x, module, scale=None, bias=None):
    if scale is not None:
        x = x * scale
    x = module(x)
    if bias is not None:
        x = x + bias
    return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

        self.gate = torch.nn.Parameter(torch.zeros(1, args.n_heads, 1, 1))
        self.head_start = self.n_local_heads * fs_init.get_model_parallel_rank()
        self.head_end = self.n_local_heads * (fs_init.get_model_parallel_rank() + 1)

        if args.add_bias:
            self.wq_bias, self.wk_bias, self.wv_bias = [
                nn.Parameter(torch.zeros([self.n_local_heads * self.head_dim])) for _ in range(3)
            ]
            self.wo_bias = nn.Parameter(torch.zeros([args.dim]))
        else:
            self.wq_bias = self.wk_bias = self.wv_bias = self.wo_bias = None

        if args.add_scale:
            self.wq_scale, self.wk_scale, self.wv_scale = [
                nn.Parameter(torch.ones([args.dim])) for _ in range(3)
            ]
            self.wo_scale = nn.Parameter(torch.ones([self.n_local_heads * self.head_dim]))
        else:
            self.wq_scale = self.wk_scale = self.wv_scale = self.wo_scale = None
        
        self.use_lora = args.use_lora
        if args.use_lora:
           self.lora_wq_l1 = nn.Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wq_l2 = nn.Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wk_l1 = nn.Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wk_l2 = nn.Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wv_l1 = nn.Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wv_l2 = nn.Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wo_l1 = nn.Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wo_l2 = nn.Linear(args.lora_rank, args.dim, bias=False)

           nn.init.constant_(self.lora_wq_l2.weight.data, 0)
           nn.init.constant_(self.lora_wk_l2.weight.data, 0)
           nn.init.constant_(self.lora_wv_l2.weight.data, 0)
           nn.init.constant_(self.lora_wo_l2.weight.data, 0)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        bsz, seqlen, _ = x.shape
        xq = forward_linear_with_scale_and_bias(x, self.wq, self.wq_scale, self.wq_bias)
        xk = forward_linear_with_scale_and_bias(x, self.wk, self.wk_scale, self.wk_bias)
        xv = forward_linear_with_scale_and_bias(x, self.wv, self.wv_scale, self.wv_bias)

        if self.use_lora:
           xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
           xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
           xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        if adapter is not None:
           adapter_len = adapter.shape[1]
           adapter_k = forward_linear_with_scale_and_bias(adapter, self.wk, self.wk_scale, self.wk_bias)
           adapter_k = adapter_k.view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
           adapter_v = forward_linear_with_scale_and_bias(adapter, self.wv, self.wv_scale, self.wv_bias)
           adapter_v = adapter_v.view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
           adapter_k = adapter_k.transpose(1, 2)
           adapter_v = adapter_v.transpose(1, 2)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        output = self._forward_scaled_dot_product_attention(xq, keys, values, mask)

        if adapter is not None:
            output += self.gate[
                :, self.head_start : self.head_end
            ].tanh().half() * self._forward_scaled_dot_product_attention(xq, adapter_k, adapter_v)

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return forward_linear_with_scale_and_bias(output, self.wo, self.wo_scale, self.wo_bias)

    def _forward_scaled_dot_product_attention(self, q, k, v, mask=None):
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, mask >= 0 if mask is not None else None)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)  # (bs, n_local_heads, slen, head_dim)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        args: ModelArgs,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        mp_size = fs_init.get_model_parallel_world_size()

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

        if args.add_bias:
            self.w1_bias, self.w3_bias = [nn.Parameter(torch.zeros([hidden_dim // mp_size])) for _ in range(2)]
            self.w2_bias = nn.Parameter(torch.zeros([dim]))
        else:
            self.w1_bias = self.w2_bias = self.w3_bias = None

        if args.add_scale:
            self.w1_scale, self.w3_scale = [nn.Parameter(torch.ones([dim])) for _ in range(2)]
            self.w2_scale = nn.Parameter(torch.ones([hidden_dim // mp_size]))
        else:
            self.w1_scale = self.w2_scale = self.w3_scale = None

        self.use_lora = args.use_lora
        if args.use_lora:
           self.lora_w1_l1 = nn.Linear(dim, args.lora_rank, bias=False)
           self.lora_w1_l2 = nn.Linear(args.lora_rank, hidden_dim, bias=False)
           self.lora_w2_l1 = nn.Linear(hidden_dim, args.lora_rank, bias=False)
           self.lora_w2_l2 = nn.Linear(args.lora_rank, dim, bias=False)
           self.lora_w3_l1 = nn.Linear(dim, args.lora_rank, bias=False)
           self.lora_w3_l2 = nn.Linear(args.lora_rank, hidden_dim, bias=False)
           nn.init.constant_(self.lora_w1_l2.weight.data, 0)
           nn.init.constant_(self.lora_w2_l2.weight.data, 0)
           nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x):
        xw1 = forward_linear_with_scale_and_bias(x, self.w1, self.w1_scale, self.w1_bias)
        xw3 = forward_linear_with_scale_and_bias(x, self.w3, self.w3_scale, self.w3_bias)
        if self.use_lora:
            xw1 = xw1 + self.lora_w1_l2(self.lora_w1_l1(x))
            xw3 = xw3 + self.lora_w3_l2(self.lora_w3_l1(x))

        act = F.silu(xw1) * xw3
        out = forward_linear_with_scale_and_bias(act, self.w2, self.w2_scale, self.w2_bias)
        if self.use_lora:
            out = out + self.lora_w2_l2(self.lora_w2_l1(act))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            args=args,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def _compute_hidden(self, tokens: torch.Tensor, start_pos: int, visual_tokens: torch.Tensor = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        adapter = self.adapter_query.weight.reshape(self.adapter_layer, self.adapter_len, self.params.dim).unsqueeze(1)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for i, layer in enumerate(self.layers):
            adapter_index = i - (len(self.layers) - self.adapter_layer)
            adapter_per_layer = adapter[adapter_index] if adapter_index >= 0 else None
            if visual_tokens is not None:
                if i in self.params.vision_early_fusion:
                    adapter_per_layer = visual_tokens
                elif adapter_per_layer is not None:
                    adapter_per_layer += visual_tokens
            if adapter_per_layer.shape[0] == 1:
                adapter_per_layer = adapter_per_layer.repeat(_bsz, 1, 1)
            h = layer(h, start_pos, freqs_cis, mask, adapter_per_layer)

        h = self.norm(h)
        return h

    def forward_train(self, tokens, visual_tokens, labels):
        h = self._compute_hidden(tokens=tokens, start_pos=0, visual_tokens=visual_tokens)
        output = self.output(h)

        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        loss = self.loss_fn(output, labels)
        return loss

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, visual_tokens: torch.Tensor = None):
        h = self._compute_hidden(tokens=tokens, start_pos=start_pos, visual_tokens=visual_tokens)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()


class VisionModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params

        self.clip, self.clip_transform = clip.load(params.vision_clip_model)
        self.clip.float()
        for param in self.clip.parameters():
            param.requires_grad = False

        self.clip_proj = nn.Linear(self.clip.visual.output_dim, params.vision_dim)
        self.clip_proj_norm = nn.LayerNorm(params.vision_dim)

        self.visual_query = nn.Embedding(params.adapter_len, params.vision_dim)

        self.visual_blocks = nn.ModuleList([
            nn.MultiheadAttention(params.vision_dim, 16, add_bias_kv=True, batch_first=True)
        for _ in range(params.vision_blocks)])

        self.visual_proj = nn.Linear(params.vision_dim, params.dim)
        self.visual_proj_norm = nn.LayerNorm(params.dim)

    def clip_encode_image(self, x: torch.Tensor):
        x = x.to(self.clip.visual.conv1.weight.dtype)
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward(self, imgs):
        if torch.is_tensor(imgs):
            x = imgs.to(self.visual_query.weight.device)
        else:
            x = [img if torch.is_tensor(img) else self.clip_transform(img) for img in imgs]
            x = torch.stack(x, dim=0).to(self.visual_query.weight.device)
        _bsz = x.shape[0]

        visual_feats = self.clip_encode_image(x).half()
        visual_feats = self.clip_proj(visual_feats)
        visual_feats = self.clip_proj_norm(visual_feats)

        visual_query = self.visual_query.weight.unsqueeze(0).repeat(_bsz, 1, 1)
        for block in self.visual_blocks:
            visual_query, _ = block(query=visual_query, key=visual_feats, value=visual_feats)

        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query
