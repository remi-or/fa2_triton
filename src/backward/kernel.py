import triton
import triton.language as tl

import math
from src.backward.compute_dkdv import _compute_column_blocks_dkdv
from src.backward.compute_dq import _compute_row_blocks_dq

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M1": 128, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M1": 128, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 128}, num_warps=8, num_stages=1),
#         # triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=4, num_stages=1), # TODO: change seqlen rounded
#         # triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=1),
#         # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
#         # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M1": lambda args: args["seqlen_q"] % args["BLOCK_M1"] == 0,
        "EVEN_N1": lambda args: args["seqlen_k"] % args["BLOCK_N1"] == 0,
        "HEADS_PADDED": lambda args: args["headdim"] != args["BLOCK_HEADDIM"],
        "NUM_BLOCKS_KV": lambda args: math.ceil(args["seqlen_k"] /  args["BLOCK_N1"]),
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
        pad_cols = (not EVEN_N1) or (VARLEN and (i_start_n * BLOCK_N1 > actual_seqlen_k)) # this works while other bools fail. Why?
        _compute_column_blocks_dkdv(
            i_start_n * BLOCK_N1,
            Q, K, V, DO, DK, DV, LSE, D,
            softmax_scale,
            stride_qm, stride_kn, stride_vn, stride_dom, stride_dkn, stride_dvn,
            actual_seqlen_q, actual_seqlen_k, headdim,
            IS_CAUSAL=IS_CAUSAL,
            PAD_COLS=True, HEADS_PADDED=HEADS_PADDED,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

    # Case: this block works on dq      
    else:
        i_start_m = pid - NUM_BLOCKS_KV
        _compute_row_blocks_dq(
            i_start_m * BLOCK_M2,
            Q, K, V, DO, DQ, LSE, D,
            softmax_scale,
            stride_qm, stride_kn, stride_vn, stride_dom, stride_dqm, 
            actual_seqlen_q, actual_seqlen_k, headdim, 
            IS_CAUSAL=IS_CAUSAL,
            PAD_ROWS=True, HEADS_PADDED=HEADS_PADDED,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_HEADDIM=BLOCK_HEADDIM,
        )