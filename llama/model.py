# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

import clip
from timm.models.vision_transformer import Block


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
    adapter_layer: int = 31

    w_bias: bool = False
    w_lora: bool = False
    lora_rank: int = 16

    v_clip_model: str = "ViT-L/14"
    v_embed_dim: int = 768
    v_depth: int = 8
    v_num_heads: int = 16
    v_mlp_ratio: float = 4.0
    v_truncate_query: bool = True
    v_early_fusion: set = field(default_factory=set)

    is_training: bool = False


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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias,
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
            bias=args.w_bias,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        if args.is_training:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()

        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

        self.use_lora = args.w_lora
        if args.w_lora:
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

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.use_lora:
           xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
           xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
           xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.cache_k is not None and self.cache_v is not None:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        if adapter is not None:
           adapter_len = adapter.shape[1]
           adapter_k = self.wk(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
           adapter_v = self.wv(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
           adapter_k = adapter_k.transpose(1, 2)
           adapter_v = adapter_v.transpose(1, 2)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        output = self._forward_scaled_dot_product_attention(xq, keys, values, mask=mask)

        if adapter is not None:
            output = output + self._forward_scaled_dot_product_attention(xq, adapter_k, adapter_v, gate=self.gate)

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        if self.use_lora:
           return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
           return self.wo(output)

    def _forward_scaled_dot_product_attention(self, q, k, v, mask=None, gate=None):
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        if gate is not None:
            scores = gate.tanh().half() * scores
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

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=args.w_bias, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=args.w_bias, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=args.w_bias, gather_output=False, init_method=lambda x: x
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)

        self.use_lora = args.w_lora
        if args.w_lora:
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
        xw1, xw3 = self.w1(x), self.w3(x)
        if self.use_lora:
            xw1 = xw1 + self.lora_w1_l2(self.lora_w1_l1(x))
            xw3 = xw3 + self.lora_w3_l2(self.lora_w3_l1(x))

        out = F.silu(xw1) * xw3
        if self.use_lora:
            return self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))
        else:
            return self.w2(out)


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
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for i, layer in enumerate(self.layers):
            adapter_index = i - (len(self.layers) - self.adapter_layer)
            adapter_per_layer = adapter[adapter_index] if adapter_index >= 0 else None
            if adapter_per_layer is not None:
                adapter_per_layer = adapter_per_layer.repeat(_bsz, 1, 1)
            if visual_tokens is not None:
                if i in self.params.v_early_fusion:
                    adapter_per_layer = visual_tokens
                elif adapter_per_layer is not None:
                    adapter_per_layer += visual_tokens
            h = layer(h, start_pos, freqs_cis, mask, adapter_per_layer)

        h = self.norm(h)
        return h

    def forward(self, tokens, visual_tokens, labels):
        h = self._compute_hidden(tokens=tokens, start_pos=0, visual_tokens=visual_tokens)
        output = self.output(h)

        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        loss = self.loss_fn(output, labels)
        return loss

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, visual_tokens: torch.Tensor = None):
        h = self._compute_hidden(tokens=tokens, start_pos=start_pos, visual_tokens=visual_tokens)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()


class VisionModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params

        self.clip, self.clip_transform = clip.load(params.v_clip_model)
        self.clip.float()
        for param in self.clip.parameters():
            param.requires_grad = False

        self.clip_proj = nn.Linear(self.clip.visual.output_dim, params.v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(params.v_embed_dim)

        self.visual_query = nn.Embedding(params.adapter_len, params.v_embed_dim)

        self.adapter_len = params.adapter_len
        self.v_truncate_query = params.v_truncate_query
        if params.v_truncate_query:
            self.visual_blocks = nn.ModuleList([
                Block(params.v_embed_dim, params.v_num_heads, params.v_mlp_ratio, qkv_bias=True)
                for _ in range(params.v_depth)])
        else:
            self.visual_blocks = nn.ModuleList([
                nn.MultiheadAttention(params.v_embed_dim, params.v_num_heads, add_bias_kv=True, batch_first=True)
                for _ in range(params.v_depth)])

        self.visual_proj = nn.Linear(params.v_embed_dim, params.dim)
        self.visual_proj_norm = nn.LayerNorm(params.dim)

    def clip_encode_image(self, x: torch.Tensor):
        x = x.type_as(self.clip.visual.conv1.weight)
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

    def forward_visual(self, x):
        _bsz = x.shape[0]

        visual_feats = self.clip_encode_image(x).float()
        visual_feats = self.clip_proj(visual_feats)
        visual_feats = self.clip_proj_norm(visual_feats)

        visual_query = self.visual_query.weight.unsqueeze(0).repeat(_bsz, 1, 1)

        if self.v_truncate_query:
            visual_query = torch.cat([visual_query, visual_feats], dim=1)
            for block in self.visual_blocks:
                visual_query = block(visual_query)
            visual_query = visual_query[:, :self.adapter_len, :]
            visual_query = self.visual_proj(visual_query)
            visual_query = self.visual_proj_norm(visual_query)
        else:
            for block in self.visual_blocks:
                visual_query, _ = block(query=visual_query, key=visual_feats, value=visual_feats)
            visual_query = self.visual_proj(visual_query)
            visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward(self, imgs):
        if torch.is_tensor(imgs):
            x = imgs.to(self.visual_query.weight.device)
        else:
            x = [img if torch.is_tensor(img) else self.clip_transform(img) for img in imgs]
            x = torch.stack(x, dim=0).to(self.visual_query.weight.device)
        return self.forward_visual(x)

    @torch.inference_mode()
    def forward_inference(self, imgs):
        x = [self.clip_transform(img) for img in imgs]
        x = torch.stack(x, dim=0)
        return self.forward_visual(x)
