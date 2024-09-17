import torch
import triton
import triton.language as tl

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

@triton.jit
def load_fn(
    ptrs, 
    offs_axis_0: tl.const_pointer_type,
    offs_axis_1: tl.const_pointer_type,
    PAD_AXIS_0: tl.constexpr,
    PAD_AXIS_1: tl.constexpr,
    LIM_AXIS_0: tl.constexpr,
    LIM_AXIS_1: tl.constexpr,
):
    if PAD_AXIS_0 and not PAD_AXIS_1: # rows only are padded
        x = tl.load(ptrs, mask=offs_axis_0[:, None] < LIM_AXIS_0, other=0.0)
    elif PAD_AXIS_0: # rows and heads are padded 
        x = tl.load(ptrs, mask=(offs_axis_0[:, None] < LIM_AXIS_0) & (offs_axis_1[None, :] < LIM_AXIS_1), other=0.0)
    elif not PAD_AXIS_1: # nothing is padded
        x = tl.load(ptrs)
    else: # only heads are padded
        x = tl.load(ptrs, mask=offs_axis_1[None, :] < LIM_AXIS_1, other=0.0)
    return x

@triton.jit
def store_fn(
    ptrs, 
    values,
    offs_axis_0: tl.const_pointer_type,
    offs_axis_1: tl.const_pointer_type,
    PAD_AXIS_0: tl.constexpr,
    PAD_AXIS_1: tl.constexpr,
    LIM_AXIS_0: tl.constexpr,
    LIM_AXIS_1: tl.constexpr,
):
    if PAD_AXIS_0 and not PAD_AXIS_1: # rows only are padded
        x = tl.store(ptrs, values, mask=offs_axis_0[:, None] < LIM_AXIS_0)
    elif PAD_AXIS_0: # rows and heads are padded 
        x = tl.store(ptrs, values, mask=(offs_axis_0[:, None] < LIM_AXIS_0) & (offs_axis_1[None, :] < LIM_AXIS_1))
    elif not PAD_AXIS_1: # nothing is padded
        x = tl.store(ptrs, values)
    else: # only heads are padded
        x = tl.store(ptrs, values, mask=offs_axis_1[None, :] < LIM_AXIS_1)
    return x