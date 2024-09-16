import triton
import triton.language as tl

from src.backward.compute_delta import _compute_delta
from src.backward.compute_column_blocks import _compute_column_block

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
#             num_warps=8,
#             num_stages=1,
#         ),
#         triton.Config(
#             {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True},
#             num_warps=8,
#             num_stages=1,
#         ),
#         # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
#         # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
#     ],
#     key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "HEADS_NOT_PADDED": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
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
    stride_bb, stride_bh, stride_bm, 
    stride_dob, stride_doh, stride_dom, 
    stride_dqb, stride_dqh, stride_dqm, 
    stride_dkb, stride_dkh, stride_dkn, 
    stride_dvb, stride_dvh, stride_dvn, 
    nheads,
    VARLEN: tl.constexpr,
    seqlen_q,
    cum_seqlens_q,
    seqlen_k,
    cum_seqlens_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    HEADS_NOT_PADDED: tl.constexpr,
):
    # Locate kernel inside the grid
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

    # if BIAS_TYPE != "none": TODO: support bias
    #     Bias += off_batch * stride_bb + off_head * stride_bh

    # Offset vector pointers for batch and head
    D += off_head_and_batch * seqlen_q_rounded
    LSE += off_head_and_batch * seqlen_q_rounded
    
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _compute_column_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                VARLEN,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                HEADS_NOT_PADDED=HEADS_NOT_PADDED,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _compute_column_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            VARLEN,
            actual_seqlen_q,
            actual_seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            HEADS_NOT_PADDED=HEADS_NOT_PADDED,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )