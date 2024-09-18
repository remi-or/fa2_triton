import triton
import triton.language as tl

from src.utils import load_fn

@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    start_n,
    BLOCK_N,
    actual_seqlen_k,
    headdim,
    VARLEN: tl.constexpr,
    EVEN_N: tl.constexpr,
    HEADS_NOT_PADDED: tl.constexpr,
):
    pad_cols = not EVEN_N or (VARLEN and ((actual_seqlen_k - start_n) < BLOCK_N))
    tl.debug_barrier()
    if pad_cols and HEADS_NOT_PADDED:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < actual_seqlen_k)
        tl.debug_barrier()
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < actual_seqlen_k)
    elif HEADS_NOT_PADDED:
        tl.store(dv_ptrs, dv)
        tl.debug_barrier()
        tl.store(dk_ptrs, dk)
    elif pad_cols:
        tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim))
        tl.debug_barrier()
        tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim))
    else:
        tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
        tl.debug_barrier()
        tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    tl.debug_barrier()


@triton.jit
def _compute_column_block(
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
    VARLEN: tl.constexpr,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Infer the starting row (which is a multiple of BLOCK_M, not BLOCK_N)
    if IS_CAUSAL:
        begin_m = start_n * BLOCK_N + actual_seqlen_q - actual_seqlen_k
        begin_m = (begin_m // BLOCK_M) * BLOCK_M if begin_m > 0 else 0
        fully_masked_lines = actual_seqlen_q - actual_seqlen_k
    else:
        begin_m = 0
        fully_masked_lines = 0
    # Since we are in a grid dimensionned to fit max_seqlen_q, some blocks may exist early
    if begin_m >= actual_seqlen_q:
        return 
    
    tl.debug_barrier()
    # Initialize offsets 
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    tl.debug_barrier()

    # Initialize value-related pointer (not stats-related)
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])

    # Bias pointer
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    tl.debug_barrier()

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # Check if we can load a whole block of K and V
    pad_cols = not EVEN_N or (VARLEN and ((actual_seqlen_k - start_n) < BLOCK_N))
    tl.debug_barrier()
    # Load K and V, which will stay in SRAM for the row-wise loop
    k = load_fn(
        k_ptrs, offs_n, offs_d,
        PAD_AXIS_0=pad_cols, PAD_AXIS_1=HEADS_PADDED,
        LIM_AXIS_0=actual_seqlen_k, LIM_AXIS_1=headdim,
    )    
    v = load_fn(
        v_ptrs, offs_n, offs_d,
        PAD_AXIS_0=pad_cols, PAD_AXIS_1=HEADS_PADDED,
        LIM_AXIS_0=actual_seqlen_k, LIM_AXIS_1=headdim,
    )
    tl.debug_barrier()
    
    # Loop over rows
    end_m = actual_seqlen_q
    for start_m in range(begin_m, end_m, BLOCK_M):

        tl.debug_barrier()
        # Update row variables
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        # Check if we can load a whole block of Q
        pad_rows = not EVEN_M or (VARLEN and (start_m + BLOCK_M > actual_seqlen_q))
        q = load_fn(q_ptrs, offs_m_curr, offs_d, pad_rows, HEADS_PADDED, actual_seqlen_q, headdim)

        tl.debug_barrier()
        # Recompute P_ij = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        
        tl.debug_barrier()
        # Attention mask
        if pad_cols:  
            qk = tl.where(offs_n[None, :] < actual_seqlen_k, qk, float("-inf"))
        # Causal mask
        if IS_CAUSAL:
            qk = tl.where(
                offs_m_curr[:, None] >= ((offs_n - actual_seqlen_k + actual_seqlen_q)[None, :]), 
                qk, 
                float("-inf"),
            )
        # TODO: fuse those? might not work (old com)

        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < actual_seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < actual_seqlen_q) & (offs_n[None, :] < actual_seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        

        tl.debug_barrier()
        # Load the LogSumExp and retrieve P
        lse_i = tl.load(LSE + offs_m_curr) # since lsm is padded to max_seqlen_q, should be good
        if BIAS_TYPE == "none":
            p = tl.exp2(qk * (softmax_scale * 1.44269504089) - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])

        tl.debug_barrier() 
        # Account for fully masked lines
        if fully_masked_lines > 0:
            p = tl.where(offs_m_curr[:, None] < fully_masked_lines, 0, p)

        tl.debug_barrier()
        # Load the gradient of O

        do = load_fn(do_ptrs, offs_m_curr, offs_d, pad_rows, HEADS_PADDED, actual_seqlen_q, headdim)
            
        tl.debug_barrier()
        # Compute the gradient of V
        dv += tl.dot(tl.trans(p).to(tl.float32), do.to(tl.float32))
        # HOTFIX : the to(fp32) brings extra precision but slows the kernel quite a bit. maybe remove it. 

        tl.debug_barrier() # TODO: rm?
        # Compute auxiliary gradients
        dp = tl.dot(do, tl.trans(v))

        tl.debug_barrier() # TODO: rm?
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        tl.debug_barrier() # TODO: rm?
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        tl.debug_barrier() # TODO: rm?
        dk += tl.dot(tl.trans(ds), q)

        tl.debug_barrier()
        
        # Compute dq in sequence mode
        if not ATOMIC_ADD:
            # Load current dq
            dq = load_fn(
                dq_ptrs, offs_m_curr, offs_d,
                PAD_AXIS_0=pad_rows, PAD_AXIS_1=HEADS_PADDED,
                LIM_AXIS_0=actual_seqlen_q, LIM_AXIS_1=headdim,
            )
            # Accumulate
            tl.debug_barrier()
            dq += tl.dot(ds, k)
            # Store
            if pad_rows and HEADS_PADDED:
                tl.store(dq_ptrs, dq, mask=(offs_m_curr[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim), eviction_policy="evict_last")
            elif HEADS_PADDED:
                tl.store(dq_ptrs, dq, mask=(offs_d[None, :] < headdim), eviction_policy="evict_last")
            elif pad_rows:
                tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < actual_seqlen_q, eviction_policy="evict_last")
            else:
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            tl.debug_barrier()

        # Compute dq in parallel mode
        else:  
            tl.debug_barrier()
            dq = tl.dot(ds, k)
            tl.debug_barrier()
            if pad_rows and HEADS_PADDED:
                tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim))
            elif HEADS_PADDED:
                tl.atomic_add(dq_ptrs, dq, mask=(offs_d[None, :] < headdim))
            elif pad_rows:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < actual_seqlen_q)
            else:
                tl.atomic_add(dq_ptrs, dq)
                
            tl.debug_barrier()
        # Increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm

    # write-back
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        start_n,
        BLOCK_N,
        actual_seqlen_k,
        headdim,
        VARLEN=VARLEN,
        EVEN_N=EVEN_N,
        HEADS_NOT_PADDED=not HEADS_PADDED,
    )
