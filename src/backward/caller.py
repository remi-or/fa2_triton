from typing import Tuple, Optional
import math
import torch
from torch import Tensor
import torch.nn.functional as F
import triton

from src.utils import attention_pack, attention_unpack
from src.backward.compute_delta import _compute_delta
from src.backward.kernel import _bwd_kernel


def _flash_attn_backward(
    dO: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    q: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    k: Tensor, # [batch_size, seqlen_k, num_heads, head_dim]
    v: Tensor, # [batch_size, seqlen_k, num_heads, head_dim]
    attention_mask: Optional[Tensor], # [batch_size, seqlen_qk]
    o: Tensor, # [batch_size, seqlen_q, num_heads, head_dim]
    lse: Tensor, # [batch_size, num_heads, max_seqlen_q_rounded]
    causal: bool = False, 
    softmax_scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    
    if attention_mask is not None:
        assert q.size(1) == k.size(1), "Attention mask is not supported with seqlen_q != seqlen_k"
        varlen_mode = (attention_mask.size(0) > 1)
        useless_padding = attention_mask.size(1) - attention_mask.sum(-1).max().item()
        if useless_padding > 0:
            dO = dO[:, :-useless_padding]
            q = q[:, :-useless_padding]
            k = k[:, :-useless_padding]
            v = v[:, :-useless_padding]
            attention_mask = attention_mask[:, :-useless_padding]
            o = o[:, :-useless_padding]
    else:
        varlen_mode = False
        useless_padding = 0

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

    # Prepare gradient accumulators # TODO: to simplify stuff, we initialize this to 0, but we could leave it empty -- check pre hook
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
        max_seqlen_q // 32,
        VARLEN=varlen_mode,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # Launch backward kernel
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N1"]) + triton.cdiv(seqlen_q, META["BLOCK_M2"]), 
        batch_size * num_heads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
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
        dO.stride(0), dO.stride(2), dO.stride(1), 
        dq.stride(0), dq.stride(2), dq.stride(1), 
        dk.stride(0), dk.stride(2), dk.stride(1), 
        dv.stride(0), dv.stride(2), dv.stride(1), 
        num_heads,
        seqlen_q,
        cum_seqlens_q,
        seqlen_k,
        cum_seqlens_k,
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 32,
        max_seqlen_k // 32,  # key for triton cache (limit number of compilations)
        VARLEN=varlen_mode,
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # In case of variable length mode, we need to unpack the gradients
    if varlen_mode:
        dq = attention_unpack(dq, cum_seqlens_q, batch_size, max_seqlen_q)
        dk = attention_unpack(dk, cum_seqlens_k, batch_size, max_seqlen_k)
        dv = attention_unpack(dv, cum_seqlens_k, batch_size, max_seqlen_k)
    # And add back the useless padding if there was any
    if useless_padding > 0:
        dq = F.pad(dq, (0, 0, 0, 0, 0, useless_padding))
        dk = F.pad(dk, (0, 0, 0, 0, 0, useless_padding))
        dv = F.pad(dv, (0, 0, 0, 0, 0, useless_padding))

    return dq, dk, dv