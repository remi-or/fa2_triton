import triton
import triton.language as tl

# TODO: exit causal blocks early
# TODO: check if using exp2 instead of exp leads to better results / times
# TODO: can we initialize accO to empty instead of 0? 

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=1, num_stages=1),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=1, num_stages=1),
#         triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=1, num_stages=1),
#     ],
#     key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "HEADS_NOT_PADDED": lambda args: args["headdim"] == args["BLOCK_HEADDIM"], # TODO: replace w/ HEADS_ARE_PADDED
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, # Q stride for the batch, head and sequence axis (sequence subscript is m for rows)
    stride_kb, stride_kh, stride_kn, # Same for K (sequence subscript is n for cols)
    stride_vb, stride_vh, stride_vn, # Same for V (sequence subscript is n for cols)
    stride_bb, stride_bh, stride_bm, # Same for bias (sequence subscript is m for rows)
    stride_ob, stride_oh, stride_om, # Same for O (sequence subscript is m for rows)
    nheads,
    seqlen_q,
    cum_seqlens_q,
    seqlen_k,
    max_seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    VARLEN: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    HEADS_NOT_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    softmax_scale = softmax_scale * 1.44269504089
    # Locate kernel inside the grid
    start_m = tl.program_id(0) # current block in the Q matrix
    off_head_and_batch = tl.program_id(1)
    off_batch = off_head_and_batch // nheads
    off_head = off_head_and_batch % nheads
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Infer actual sequence length of Q and the offset to the last sequence
    if VARLEN:
        actual_seqlen_q = tl.load(cum_seqlens_q + off_batch + 1) - tl.load(cum_seqlens_q + off_batch) 
        actual_seqlen_k = actual_seqlen_q # TODO: support packed + varlen? rn, check is done outside
        cu_seq_start_q = tl.load(cum_seqlens_q + off_batch) 
        cu_seq_start_k = tl.load(cum_seqlens_q + off_batch) 
        off_batch = 0
    else:
        actual_seqlen_q = seqlen_q
        actual_seqlen_k = seqlen_k
        cu_seq_start_q = 0
        cu_seq_start_k = 0

    # When in VARLEN mode, since we dimension the grid to be large enough for all sequences, the 
    # current sequence might have less rows than the current row (detemined through the grid).
    if start_m * BLOCK_M >= actual_seqlen_q:
        return
    
    fully_masked_lines = actual_seqlen_q - actual_seqlen_k if IS_CAUSAL else 0
    if fully_masked_lines > start_m * BLOCK_M:
        return

    # Check if we can load a whole block of Q or we need boundary checks
    pad_rows = not EVEN_M
    if VARLEN:
        pad_rows = pad_rows or (actual_seqlen_q - start_m) < BLOCK_M
    # TODO: add an exit point in case of a fully masked block (nans_line = BLOCK_M) (might already be the case)
        
    # Initialize pointers to Q, K, V # TODO: check if this uses int32 or int64 math (check FA repo)
    offseted_Q = Q + off_batch * stride_qb + off_head * stride_qh + cu_seq_start_q * stride_qm
    q_ptrs = (offseted_Q + (offs_m[:, None] * stride_qm + offs_d[None, :]))
    offseted_K = K + off_batch * stride_kb + off_head * stride_kh + cu_seq_start_k * stride_kn
    k_ptrs = (offseted_K + (offs_n[:, None] * stride_kn + offs_d[None, :]))
    offseted_V = V + off_batch * stride_vb + off_head * stride_vh + cu_seq_start_k * stride_vn
    v_ptrs = (offseted_V + (offs_n[:, None] * stride_vn + offs_d[None, :]))

    # Bias
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_batch * stride_bb + off_head * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_batch * stride_bb
            + off_head * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    
    # Initialize pointers to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q, which will stay in SRAM for the whole loop
    if pad_rows and HEADS_NOT_PADDED: # rows only are padded
        q = tl.load(q_ptrs, mask=offs_m[:, None] < actual_seqlen_q, other=0.0)
    elif pad_rows: # rows and heads are padded 
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    elif HEADS_NOT_PADDED: # nothing is padded
        q = tl.load(q_ptrs)
    else: # only heads are padded
        q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

    # Compute last visited column of KV which 
    if IS_CAUSAL:
        end_n = min(actual_seqlen_k - actual_seqlen_q + (start_m + 1) * BLOCK_M, actual_seqlen_k)
        # For a seqlen_q >> seqlen_k, there migh be entire block skipped
        if end_n < 0:
            return
    else:
        end_n = actual_seqlen_k

    # loop over k, v and update accumulator
    # for i_n in range(0, end_n // BLOCK_N): 
    #     start_n = ((i_n + start_m) % (end_n // BLOCK_N)) * BLOCK_N
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Check if we can load a whole block of K 
        pad_cols = not EVEN_N or (VARLEN and ((actual_seqlen_k - start_n) < BLOCK_N))
        # Load K (same mechanism as for Q, only check cols instead of rows)
        offset_k_ptrs = k_ptrs + start_n * stride_kn
        if pad_cols and HEADS_NOT_PADDED:
            k = tl.load(offset_k_ptrs, mask=(start_n + offs_n)[:, None] < actual_seqlen_k, other=0.0) 
        elif pad_cols:
            k = tl.load(
                offset_k_ptrs, 
                mask=((start_n + offs_n)[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim), 
                other=0.0,
            )
        elif HEADS_NOT_PADDED:
            k = tl.load(offset_k_ptrs)
        else:
            k = tl.load(offset_k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

        # Compute QK
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Apply attention masking and/or account for padding of the keys
        if pad_cols:  
            qk += tl.where((start_n + offs_n)[None, :] < actual_seqlen_k, 0, float("-inf"))
        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n - actual_seqlen_k + actual_seqlen_q)[None, :]
            qk += tl.where(causal_mask, 0, float("-inf"))

        # Add attention bias
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            P_ij = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            P_ij = tl.exp2(qk * softmax_scale - m_ij[:, None])

        # Accumulate stats
        l_ij = tl.sum(P_ij, 1)

        # Scale the output accumulator
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        # Load V (same mechanism as K)
        offset_v_ptrs = v_ptrs + start_n * stride_vn
        if pad_cols and HEADS_NOT_PADDED:
            v = tl.load(offset_v_ptrs, mask=(start_n + offs_n)[:, None] < actual_seqlen_k, other=0.0) 
        elif pad_cols:
            v = tl.load(
                offset_v_ptrs, 
                mask=((start_n + offs_n)[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim), 
                other=0.0,
            )
        elif HEADS_NOT_PADDED:
            v = tl.load(offset_v_ptrs)
        else:
            v = tl.load(offset_v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

        # Update the output accumulator
        P_ij = P_ij.to(v.dtype)
        acc_o += tl.dot(P_ij, v)

        # Update the statistics
        m_i = m_ij
        l_i_new = tl.exp2(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log2(l_i_new)

    # Final scaling of the output accumulator
    o_scale = tl.exp2(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    # For seqlen_q >> seqlen_k, there might be entire lines masked, so we account for that
    if fully_masked_lines > start_m * BLOCK_M:
        acc_o = tl.where(offs_m[:, None] < fully_masked_lines, 0, acc_o)

    # rematerialize offsets to save registers (?)
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Write back l and m
    ## Q + off_batch * stride_qb + off_head * stride_qh + cu_seq_start_q * stride_qm
    lse_ptrs = Lse + off_head_and_batch * max_seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i / 1.44269504089)
    # Initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_batch * stride_ob
        + off_head * stride_oh
        + cu_seq_start_q * stride_om
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    # Store O (same mechanism as Q)
    if pad_rows and HEADS_NOT_PADDED:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < actual_seqlen_q)
    elif pad_rows:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim))
    elif HEADS_NOT_PADDED: # nothing is padded
        tl.store(out_ptrs, acc_o)
    else: # only heads are padded
        tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)