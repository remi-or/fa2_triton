# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import sys 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import product

from tests.utils import generate_test_data
from other_implemenations.reference_implementation import attention_ref
from benchmarks.utils import benchmark_all, benchmark_forward, benchmark_backward, benchmark_fwd_bwd, benchmark_combined
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from src.wrapper import flash_attn_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def causal_mask_fn(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def flex_attn_func(q, k, v, attn_mask, attn_bias, causal):
    if causal:
        block_mask = create_block_mask(causal_mask_fn, B=None, H=None, Q_LEN=1024, KV_LEN=1024)
        return flex_attention(q, k, v, block_mask=block_mask)
    else:
        return flex_attention(q, k, v)

repeats = 30
warmups = 3
device = 'cuda'
dtype = torch.float16

dropout_p = 0.0

methods = (["Liger", "Flex", "Pytorch"]
        #    + (["Triton"] if attention_triton is not None else [])
           + (["xformers.c"] if xops is not None else [])
           + (["xformers.f"] if xops is not None else []))

LATENT_DIM = 2048
BATCH_SIZES_AND_SEQLENS = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
HEAD_DIMS = [64, 128]

if __name__ == "__main__":

    # Prepare accumulators
    times_fwd = {}
    times_bwd = {}
    times_both = {}
    speeds_fwd = {}
    speeds_bwd = {}
    speeds_both = {}

    # Main loop
    for (batch_size, seqlen), head_dim, causal in product(BATCH_SIZES_AND_SEQLENS, HEAD_DIMS, [False, True]):

        # Setup tests
        config = (causal, head_dim, batch_size, seqlen)
        num_heads = LATENT_DIM // head_dim
        q, k, v, _ = generate_test_data(
            batch_size=batch_size, 
            num_heads=num_heads, 
            seqlen_q=seqlen, 
            seqlen_k=seqlen,
            head_dim=head_dim,
            dtype=torch.float16,
        )

        # Time the liger implementation
        f, b = time_fwd_bwd(flash_attn_func, q, k, v, None, None, causal, repeats=repeats, warmups=warmups, verbose=False)
        times_fwd[config, "Liger"] = f
        times_bwd[config, "Liger"] = b

        # Time the flexattention implementation
        f, b = time_fwd_bwd(flex_attn_func, q, k, v, None, None, causal, repeats=repeats, warmups=warmups, verbose=False)
        times_fwd[config, "Flex"] = f
        times_bwd[config, "Flex"] = b

        # # Time the liger implementation
        # f, b = time_fwd_bwd(attention_triton, q, k, v, causal, 0.5, repeats=repeats, warmups=warmups, verbose=False)
        # times_fwd[config, "Liger"] = f
        # times_bwd[config, "Liger"] = b

        # Time the pytorch implementation
        try:
            f, b = time_fwd_bwd(attention_ref, q, k, v, causal=causal, repeats=repeats, warmups=warmups, verbose=False)
        except:  # Skip if OOM
            f, b = float('nan'), float('nan')
        times_fwd[config, "Pytorch"] = f
        times_bwd[config, "Pytorch"] = b

        # Display times
        print(f"### causal={causal}, headdim={head_dim}, batch_size={batch_size}, seqlen={seqlen} ###")
        for method in methods:
            times_both[config, method] = times_fwd[config, method] + times_bwd[config, method]
            speeds_fwd[config, method] = efficiency(
                flops(batch_size, seqlen, head_dim, num_heads, causal, mode="fwd"),
                times_fwd[config, method]
            )
            speeds_bwd[config, method] = efficiency(
                flops(batch_size, seqlen, head_dim, num_heads, causal, mode="bwd"),
                times_bwd[config, method]
            )
            speeds_both[config, method] = efficiency(
                flops(batch_size, seqlen, head_dim, num_heads, causal, mode="fwd_bwd"),
                times_both[config, method]
            )
            print(
                f"{method} fwd: {speeds_fwd[config, method]:.2f} TFLOPs/s, "
                f"bwd: {speeds_bwd[config, method]:.2f} TFLOPs/s, "
                f"fwd + bwd: {speeds_both[config, method]:.2f} TFLOPs/s"
            )

            if xops is not None:
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention, q, k, v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)
                )
                times_fwd[config, "xformers.c"] = f
                times_bwd[config, "xformers.c"] = b

            if xops is not None:
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention, q, k, v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
                )
                times_fwd[config, "xformers.f"] = f
                times_bwd[config, "xformers.f"] = b
