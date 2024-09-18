import math
from typing import Tuple, Optional

import torch
from torch import Tensor
import triton

from src.utils import attention_pack, attention_unpack
from src.forward.kernel import _fwd_kernel


def _flash_attn_forward(
    q: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    k: Tensor, # [batch_size, seqlen_k, num_heads, head_dim]
    v: Tensor, # [batch_size, seqlen_k, num_heads, head_dim]
    attention_mask: Optional[Tensor], # [batch_size, seqlen_qk]
    bias: Optional[Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, float]:
    
    # Currently, attention bias is not supported (TODO)
    if bias is not None:
        raise NotImplementedError("Attention bias is not yet supported.")
    # Currently, variable length (varlen) mode is mutually exclusive with attention masking (TODO)
    if attention_mask is not None:
        varlen_mode = True
        assert q.size(1) == k.size(1), "Attention mask is not supported with seqlen_q != seqlen_k"
    else:
        varlen_mode = False

    # Retrieve and check shapes (TODO: remove as much as possible of those)
    batch, seqlen_q, nheads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, head_dim)
    assert v.shape == (batch, seqlen_k, nheads, head_dim)
    assert head_dim <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale

    # Depending on attention_mask, switch to varlen
    varlen_mode = varlen_mode and (batch > 1)
    if varlen_mode:
        # Compute padding-related statistics
        cum_seqlens_q = torch.zeros(size=(attention_mask.size(0)+1,), device=attention_mask.device, dtype=torch.int32)
        cum_seqlens_q[1:] = attention_mask.sum(dim=1).cumsum(0) 
        # cum_seqlens_q = [0, seqlen_q1, seqlen_q1+seqlen_q2, ..., seqlen_q1+...+seqlen_qB] of shape [B+1]
        max_seqlen_q: int = attention_mask.size(1)
        max_seqlen_k: int = attention_mask.size(1)
        # Collate all matrices
        q = attention_pack(q, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        k = attention_pack(k, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        v = attention_pack(v, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        # Update seqlens
        seqlen_q = q.size(1)
    else:
        cum_seqlens_q = None
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

    # Setup output accumulator
    o = torch.zeros_like(q)

    # Bias (ignored rn)
    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    # endregion

    # Setup LSE accumulators: in varlen mode, batch is still equal to the nb of queries
    max_seqlen_q_rounded = math.ceil(max_seqlen_q / 128) * 128 # wastefull in varlen and not (just use mask)
    lse = torch.zeros((batch, nheads, max_seqlen_q_rounded), device=q.device, dtype=torch.float32)

    # Infer problem size and launch kernel
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    # BLOCK = 128
    # num_warps = 4 if head_dim <= 64 else 8
    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *bias_strides,
        o.stride(0), o.stride(2), o.stride(1),
        nheads,
        seqlen_q,
        cum_seqlens_q, # array containing [seqlen_q_1, ..., seqlen_q_B] , if VARLEN, else None
        seqlen_k,
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 128,
        max_seqlen_k // 128,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # VARLEN=varlen_mode, IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        varlen_mode,
        bias_type,
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # When in variable length mode, we need to unpack the packed tensors
    if varlen_mode:
        o = attention_unpack(o, cum_seqlens_q, *attention_mask.shape)

    return o, lse, softmax_scale  # softmax_scale could have been updated

