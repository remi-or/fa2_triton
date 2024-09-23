# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import sys 
from typing import Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda import OutOfMemoryError
try:
    from src.other_implemenations.flex_attention import flex_attention
    FLEX_AVAILABLE = True
except ModuleNotFoundError:
    FLEX_AVAILABLE = False


from tests.utils import generate_test_data, generate_attention_mask
from src.other_implemenations.reference_implementation import attention_ref
from benchmarks.bench_fns import benchmark_fwd_bwd
from src.wrapper import flash_attn_func


# Conversion functions
def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


# Timer function
def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def measure_kernel_latency(
    kernel: str,
    repeats: int,
    batch_size: int, 
    num_heads: int, 
    seqlen: int, 
    head_dim: int, 
    causal: bool, 
    use_attention: bool, 
    dtype: torch.dtype,
) -> Optional[Tuple[float, float]]:
    q, k, v, _ = generate_test_data(
        batch_size=batch_size, 
        num_heads=num_heads, 
        seqlen_q=seqlen, 
        seqlen_k=seqlen,
        head_dim=head_dim,
        dtype=dtype,
    )
    attn_mask = generate_attention_mask(q) if use_attention else None
    if kernel == "Liger":
        return time_fwd_bwd(flash_attn_func, q, k, v, attn_mask, None, causal, repeats=repeats, verbose=False)
    elif kernel == "Flex":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        try:
            return time_fwd_bwd(flex_attention, q, k, v, causal, repeats=repeats, verbose=False)
        except OutOfMemoryError:
            return None
    elif kernel == "Pytorch":
        try:
            return time_fwd_bwd(attention_ref, q, k, v, attn_mask, attn_mask, causal=causal, repeats=repeats, verbose=False)
        except OutOfMemoryError:
            return None
    
def measure_kernel_tflops(
    kernel: str,
    mode: str,
    repeats: int,
    batch_size: int, 
    num_heads: int, 
    seqlen: int, 
    head_dim: int, 
    causal: bool, 
    dtype: torch.dtype,
) -> Optional[float]:
    latency = measure_kernel_latency(
        kernel=kernel, mode=mode, repeats=repeats,
        batch_size=batch_size, num_heads=num_heads, seqlen=seqlen, head_dim=head_dim, 
        causal=causal, dtype=dtype,
    )
    if latency is None:
        return None
    else:
        return efficiency(
            flops(batch_size, seqlen, head_dim, num_heads, causal, mode=mode),
            latency,
        )