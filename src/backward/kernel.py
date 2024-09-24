import math
from typing import Any, Dict, List

import triton
import triton.language as tl
from triton import Config

from src.backward.compute_dkdv import _compute_column_blocks_dkdv
from src.backward.compute_dq import _compute_row_blocks_dq

MIN_B = 16


def early_config_prune_bwd_kernel(
    configs: List[Config],
    named_args: Dict[str, Any],
    **kwargs,
) -> List[Config]:
    # Remove the configs where BLOCK_ > seqlen_
    kept_configs = []
    for cfg in configs:
        block_m_too_large = max(cfg.kwargs["BLOCK_M1"], cfg.kwargs["BLOCK_M2"]) > named_args["seqlen_q"]
        block_n_too_large = max(cfg.kwargs["BLOCK_N1"], cfg.kwargs["BLOCK_N2"]) > named_args["seqlen_k"]
        if not (block_m_too_large or block_n_too_large):
            kept_configs.append(cfg)
    # If no config is left, go for the minimal config
    if configs:
        return configs
    return [Config({"BLOCK_M1": MIN_B, "BLOCK_N1": MIN_B, "BLOCK_M2": MIN_B, "BLOCK_N2": MIN_B}, num_warps=4, num_stages=0)]


@triton.autotune(
    configs=[
        Config({"BLOCK_M1": MIN_B, "BLOCK_N1": MIN_B, "BLOCK_M2": MIN_B, "BLOCK_N2": MIN_B}, num_warps=4, num_stages=0),
        Config({"BLOCK_M1": 32, "BLOCK_N1": 16, "BLOCK_M2": 16, "BLOCK_N2": 32}, num_warps=4, num_stages=0),
        Config({"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}, num_warps=4, num_stages=0),
        Config({"BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64}, num_warps=4, num_stages=0),
        Config({"BLOCK_M1": 64, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 64}, num_warps=4, num_stages=0),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "IS_CAUSAL", "BLOCK_HEADDIM"],  # TODO: add dtype
    prune_configs_by={"early_config_prune": early_config_prune_bwd_kernel},
)
@triton.heuristics(
    {
        "EVEN_M1": lambda args: args["seqlen_q"] % args["BLOCK_M1"] == 0,
        "EVEN_N1": lambda args: args["seqlen_k"] % args["BLOCK_N1"] == 0,
        "EVEN_M2": lambda args: args["seqlen_q"] % args["BLOCK_M2"] == 0,
        "EVEN_N2": lambda args: args["seqlen_k"] % args["BLOCK_N2"] == 0,
        "HEADS_PADDED": lambda args: args["headdim"] != args["BLOCK_HEADDIM"],
        "NUM_BLOCKS_KV": lambda args: math.ceil(args["seqlen_k"] / args["BLOCK_N1"]),
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    stride_dkb, stride_dkh, stride_dkn,
    stride_dvb, stride_dvh, stride_dvn,
    nheads,
    seqlen_q,
    cum_seqlens_q,
    seqlen_k,
    cum_seqlens_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    NUM_BLOCKS_KV: tl.constexpr,
    EVEN_M1: tl.constexpr,
    EVEN_N1: tl.constexpr,
    EVEN_M2: tl.constexpr,
    EVEN_N2: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
):
    # Locate kernel inside the grid
    pid = tl.program_id(0)
    off_head_and_batch = tl.program_id(1)
    off_batch = off_head_and_batch // nheads
    off_head = off_head_and_batch % nheads

    # If in variable length mode, retrieve the actual sequence lengths
    if VARLEN:
        cu_seq_start_q = tl.load(cum_seqlens_q + off_batch)
        cu_seq_start_k = tl.load(cum_seqlens_k + off_batch)
        actual_seqlen_q = tl.load(cum_seqlens_q + off_batch + 1) - cu_seq_start_q
        actual_seqlen_k = tl.load(cum_seqlens_k + off_batch + 1) - cu_seq_start_k
        off_batch = 0
    else:
        cu_seq_start_q = 0
        cu_seq_start_k = 0
        actual_seqlen_q = seqlen_q
        actual_seqlen_k = seqlen_k

    # Offset matrix pointers for batch and head
    Q += off_batch * stride_qb + off_head * stride_qh + cu_seq_start_q * stride_qm
    K += off_batch * stride_kb + off_head * stride_kh + cu_seq_start_k * stride_kn
    V += off_batch * stride_vb + off_head * stride_vh + cu_seq_start_k * stride_vn
    DO += off_batch * stride_dob + off_head * stride_doh + cu_seq_start_q * stride_dom
    DQ += off_batch * stride_dqb + off_head * stride_dqh + cu_seq_start_q * stride_dqm
    DK += off_batch * stride_dkb + off_head * stride_dkh + cu_seq_start_k * stride_dkn
    DV += off_batch * stride_dvb + off_head * stride_dvh + cu_seq_start_k * stride_dvn

    # Offset vector pointers for batch and head
    D += off_head_and_batch * seqlen_q_rounded
    LSE += off_head_and_batch * seqlen_q_rounded

    # Case: this block works on dk and dv
    if pid < NUM_BLOCKS_KV:
        i_start_n = pid
        pad_cols = (not EVEN_N1) or (VARLEN and ((i_start_n + 1) * BLOCK_N1 > actual_seqlen_k))
        _compute_column_blocks_dkdv(
            i_start_n * BLOCK_N1,
            Q, K, V, DO, DK, DV, LSE, D,
            softmax_scale,
            stride_qm, stride_kn, stride_vn, stride_dom, stride_dkn, stride_dvn,
            actual_seqlen_q, actual_seqlen_k, headdim,
            IS_CAUSAL=IS_CAUSAL,
            PAD_COLS=pad_cols, HEADS_PADDED=HEADS_PADDED,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

    # Case: this block works on dq
    else:
        i_start_m = pid - NUM_BLOCKS_KV
        pad_rows = (not EVEN_M2) or (VARLEN and ((i_start_m + 1) * BLOCK_M2 > actual_seqlen_q))
        _compute_row_blocks_dq(
            i_start_m * BLOCK_M2,
            Q, K, V, DO, DQ, LSE, D,
            softmax_scale,
            stride_qm, stride_kn, stride_vn, stride_dom, stride_dqm,
            actual_seqlen_q, actual_seqlen_k, headdim,
            VARLEN=VARLEN, IS_CAUSAL=IS_CAUSAL,
            PAD_ROWS=pad_rows, HEADS_PADDED=HEADS_PADDED,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_N=EVEN_N2,
        )
