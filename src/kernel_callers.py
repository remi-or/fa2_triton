"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math
from typing import Tuple, Optional

import torch
from torch import Tensor
import triton

from src.forward.kernel import _fwd_kernel
from src.backward.compute_delta import _compute_delta
from src.backward.kernel import _bwd_kernel

DEBUG_DELTA = [False]


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )

def attention_pack(
    x: torch.Tensor, # [batch_size, seqlen, num_heads, head_dim]
    attention_mask: torch.Tensor, # [batch_size, seqlen]
) -> torch.Tensor:
    to_pack = [] 
    for i, attn_mask in enumerate(attention_mask):
        seqlen = attn_mask.sum().int().item()
        kept = x[i, :seqlen] # [seqlen, num_heads, head_dim]
        to_pack.append(kept)
    return torch.concatenate(to_pack, dim=0).unsqueeze(0)

def attention_unpack(
    x: torch.Tensor, # [1, sum_seqlens, num_heads, head_dim]
    cum_seqlens: torch.Tensor, # [0, seqlen_1, seqlen_1+seqlen_2, ...]
    batch_size: int,
    goal_seqlen: int,
) -> torch.Tensor:
    unpacked = torch.zeros(size=(batch_size, goal_seqlen, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(cum_seqlens.size(0)-1):
        seq_start = cum_seqlens[i]
        seq_end = cum_seqlens[i+1]
        unpacked[i, :seq_end-seq_start] = x[0, seq_start:seq_end]
    return unpacked


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
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = 1.0 / math.sqrt(d) if softmax_scale is None else softmax_scale

    # Depending on attention_mask, switch to varlen
    if varlen_mode:
        # Compute padding-related statistics
        cum_seqlens_q = torch.zeros(size=(attention_mask.size(0)+1,), device=attention_mask.device, dtype=torch.int32)
        cum_seqlens_q[1:] = attention_mask.sum(dim=1).cumsum(0) 
        # cum_seqlens_q = [0, seqlen_q1, seqlen_q1+seqlen_q2, ..., seqlen_q1+...+seqlen_qB] of shape [B+1]
        max_seqlen_q: int = attention_mask.size(1)
        # Collate all matrices
        q = attention_pack(q, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        k = attention_pack(k, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        v = attention_pack(v, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        # Update seqlens
        seqlen_q = q.size(1)
    else:
        cum_seqlens_q = None
        max_seqlen_q = seqlen_q

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
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
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
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # VARLEN=varlen_mode, IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        varlen_mode,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    # When in variable length mode, we need to unpack the packed tensors
    if varlen_mode:
        o = attention_unpack(o, cum_seqlens_q, *attention_mask.shape)

    return o, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attn_backward(
    dO: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    q: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    k: Tensor, # [batch_size, seqlen_k, num_heads, head_dim]
    v: Tensor, # [batch_size, seqlen_k, num_heads, head_dim]
    attention_mask: Optional[Tensor], # [batch_size, seqlen_qk]
    o: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    lse: Tensor, # [batch_size, seqlen_q, num_heads]
    bias: Optional[Tensor] = None, 
    causal: bool = False, 
    softmax_scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    
    # Currently, we do not allow for attention bias
    if bias is not None:
        raise NotImplementedError("Attention bias is not supported yet")
    # Currently, variable length (varlen) mode is mutually exclusive with attention masking (TODO)
    varlen_mode = (attention_mask is not None)
    if varlen_mode:
        assert q.size(1) == k.size(1), "Attention mask is not supported with seqlen_q != seqlen_k"

    # Retrieve and check shapes
    dO = dO.contiguous() if dO.stride(-1) != 1 else dO
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k = k.size(1)
    max_seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    assert head_dim <= 128
    assert lse.shape == (batch_size, num_heads, max_seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1

    # Depending on attention_mask, switch to varlen
    if varlen_mode:
        # TODO: right now, seqlen is identical between Q and K. as that might change, we leave the doubles for later use
        # Compute padding-related statistics
        cum_seqlens_q = torch.zeros(size=(attention_mask.size(0)+1,), device=attention_mask.device, dtype=torch.int32)
        cum_seqlens_q[1:] = attention_mask.sum(dim=1).cumsum(0) 
        cum_seqlens_k = torch.zeros(size=(attention_mask.size(0)+1,), device=attention_mask.device, dtype=torch.int32)
        cum_seqlens_k[1:] = attention_mask.sum(dim=1).cumsum(0) 
        # cum_seqlens_q = [0, seqlen_q1, seqlen_q1+seqlen_q2, ..., seqlen_q1+...+seqlen_qB] of shape [B+1]
        max_seqlen_q: int = attention_mask.size(1)
        max_seqlen_k: int = attention_mask.size(1)
        # Collate all matrices
        q = attention_pack(q, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        k = attention_pack(k, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        v = attention_pack(v, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        o = attention_pack(o, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        dO = attention_pack(dO, attention_mask) # [1, sum_seqlens_qk, num_head, head_dim]
        # Update seqlens
        seqlen_q = q.size(1)
        seqlen_k = k.size(1)
    else:
        cum_seqlens_q = None
        cum_seqlens_k = None
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

    # Prepare gradient accumulators # TODO: to simplify stuff, we initialize this to 0, but we could leave it empty
    dq = torch.zeros_like(q, dtype=torch.float32) # [batch_size|1, seqlen_q|sum_seqlens_qk, num_heads, head_dim]
    dk = torch.zeros_like(k) # [batch_size|1, seqlen_q|sum_seqlens_q, num_heads, head_dim]
    dv = torch.zeros_like(v) # [batch_size|1, seqlen_q|sum_seqlens_k, num_heads, head_dim]
    delta = torch.zeros_like(lse) # [batch_size, num_heads, max_seqlen_q_rounded]

    # Infer problem size
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    # Launch the delta computation kernel
    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch_size * num_heads)
    _compute_delta[grid](
        o,
        dO,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        dO.stride(0),
        dO.stride(2),
        dO.stride(1),
        num_heads,
        seqlen_q,
        max_seqlen_q_rounded,
        cum_seqlens_q,
        head_dim,
        VARLEN=varlen_mode,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    if DEBUG_DELTA[0]:
        print(delta.shape)
        delta = delta[:, :, :max_seqlen_q].transpose(1, 2).unsqueeze(-1)
        print(delta.shape)
        delta = delta.expand((*delta.shape[:-1], head_dim))
        print(delta.shape)
        return delta, None, None
    

    # Bias
    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch_size, num_heads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4

    # Launch backward kernel
    grid = lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]), batch_size * num_heads)
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        dO,
        dq,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1), 
        k.stride(0), k.stride(2), k.stride(1), 
        v.stride(0), v.stride(2), v.stride(1), 
        *bias_strides,
        dO.stride(0), dO.stride(2), dO.stride(1), 
        dq.stride(0), dq.stride(2), dq.stride(1), 
        dk.stride(0), dk.stride(2), dk.stride(1), 
        dv.stride(0), dv.stride(2), dv.stride(1), 
        num_heads,
        varlen_mode,
        seqlen_q,
        cum_seqlens_q,
        seqlen_k,
        cum_seqlens_k,
        max_seqlen_q_rounded,
        head_dim,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        True, # SEQUENCE_PARALLEL # TODO: check what happens when False
        128, # BLOCK_M
        128, # BLOCK_N
        # SEQUENCE_PARALLEL=False,  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  num_warps=num_warps, num_stages=1,
    )

    # In case of variable length mode, we need to unpack the gradients
    if varlen_mode:
        dq = attention_unpack(dq, cum_seqlens_q, batch_size, max_seqlen_q)
        dk = attention_unpack(dk, cum_seqlens_k, batch_size, max_seqlen_k)
        dv = attention_unpack(dv, cum_seqlens_k, batch_size, max_seqlen_k)

    return dq, dk, dv