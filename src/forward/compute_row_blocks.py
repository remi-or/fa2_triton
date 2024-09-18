import triton
import triton.language as tl

from src.utils import load_fn

@triton.jit
def compute_row_block(
    q,
    m_i,
    lse_i,
    k_ptrs,
    v_ptrs,
    acc_o,
    offs_m,
    offs_n,
    offs_d,
    softmax_scale,
    stride_kn,
    stride_vn,
    start_n,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    IS_CAUSAL: tl.constexpr,
    MASKED: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)

    # Check if we can load a whole block of K
    # if actual_seqlen_k % BLOCK_N == 0:
    #     pad_cols = ((actual_seqlen_k - start_n) < BLOCK_N) and last_block
    # else:
    #     pad_cols = last_block
    pad_cols = True # TODO: fid fix

    # Load K (same mechanism as for Q, only check cols instead of rows)
    offset_k_ptrs = k_ptrs + start_n * stride_kn
    k = load_fn(
        offset_k_ptrs, 
        start_n + offs_n, offs_d, 
        PAD_AXIS_0=pad_cols, PAD_AXIS_1=PADDED_HEADS, 
        LIM_AXIS_0=actual_seqlen_k, LIM_AXIS_1=headdim,
    )

    # Compute QK
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))

    # Apply attention masking and/or account for padding of the keys
    if pad_cols:  
        qk += tl.where((start_n + offs_n)[None, :] < actual_seqlen_k, 0, float("-inf"))
    # Apply causal mask
    if MASKED and IS_CAUSAL:
        causal_mask = offs_m[:, None] >= (start_n + offs_n - actual_seqlen_k + actual_seqlen_q)[None, :]
        qk += tl.where(causal_mask, 0, float("-inf"))

    # else:
    m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
    P_ij = tl.exp2(qk * softmax_scale - m_ij[:, None])

    # Accumulate stats
    l_ij = tl.sum(P_ij, 1)

    # Scale the output accumulator
    acc_o_scale = tl.exp2(m_i - m_ij)
    acc_o = acc_o * acc_o_scale[:, None]

    # Load V (same mechanism as K)
    offset_v_ptrs = v_ptrs + start_n * stride_vn
    v = load_fn(
        offset_v_ptrs, 
        start_n + offs_n, offs_d, 
        PAD_AXIS_0=pad_cols, PAD_AXIS_1=PADDED_HEADS, 
        LIM_AXIS_0=actual_seqlen_k, LIM_AXIS_1=headdim,
    )

    # Update the output accumulator
    P_ij = P_ij.to(v.dtype)
    acc_o += tl.dot(P_ij, v)

    # Update the statistics
    m_i = m_ij
    l_i_new = tl.exp2(lse_i - m_ij) + l_ij
    lse_i = m_ij + tl.log2(l_i_new)

    return m_i, lse_i, acc_o