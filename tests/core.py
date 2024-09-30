import torch
from torch import Tensor
from typing import Optional, Tuple

from src.reference_implementation import flash_attn_reference
from src.wrapper import flash_attn_func
from tests.utils import compare_results_fa, generate_attention_mask, generate_test_data, generate_dropout_seed_and_mask


def _test_core_fn(
    batch_size: int,
    nheads_q: int,
    nheads_kv: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    causal: bool,
    dropout_p: float,
    use_attention: bool,
    use_bias: bool,
    dtype: torch.dtype,
    FORWARD_ONLY: bool,
    RETURN: bool,
) -> Optional[Tuple[Tensor, ...]]:
    # Prepare data
    q, k, v, do = generate_test_data(batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k, head_dim, dtype)
    attn_mask = generate_attention_mask(q) if use_attention else None
    attn_bias = torch.rand(size=(1, 1, seqlen_q, seqlen_k), dtype=dtype, device=q.device) if use_bias else None
    dropout_seed, dropout_mask = generate_dropout_seed_and_mask(dropout_p, q, k, attn_mask)
    # Compute reference
    out_ref = flash_attn_reference(
        q=q,
        k=k,
        v=v,
        query_padding_mask=attn_mask,
        key_padding_mask=attn_mask,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        dropout_mask=dropout_mask,
        causal=causal,
    )
    # Compute pytorch reference
    out_pt = flash_attn_reference(
        q=q,
        k=k,
        v=v,
        query_padding_mask=attn_mask,
        key_padding_mask=attn_mask,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        dropout_mask=dropout_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True
    )
    # Compute ours
    out = flash_attn_func(
        q=q,
        k=k,
        v=v,
        attention_mask=attn_mask,
        attention_bias=attn_bias,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=None,
        dropout_seed=dropout_seed,
    )
    # Compare results
    grad = None if FORWARD_ONLY else do
    if RETURN:
        try:
            compare_results_fa(q, k, v, grad, out, out_ref, out_pt)
        except AssertionError as ae:
            print(ae)
        return q, k, v, out, out_pt, out_ref, do
    else:
        compare_results_fa(q, k, v, grad, out, out_ref, out_pt)
        return None
