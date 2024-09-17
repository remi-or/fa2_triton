"""Here, we test if the kernel when there is no attention mask and seqlen_q != seqlen_k."""

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

import torch
import pytest

from src.wrapper import flash_attn_func
from tests.utils import generate_test_data, compare_results_fa, generate_attention_mask
from src.other_implemenations.reference_implementation import attention_ref


def _test_fwd_bwd(
    batch_size: int, 
    num_heads: int, 
    seqlen_q: int, 
    seqlen_k: int, 
    head_dim: int, 
    causal: bool, 
    use_attention: bool, 
    dtype: torch.dtype = torch.float16,
) -> None: 
    # Prepare data
    q, k, v, do = generate_test_data(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
    attn_mask = generate_attention_mask(q) if use_attention else None
    # Compute reference
    out_ref = attention_ref(q, k, v, query_padding_mask=attn_mask, key_padding_mask=attn_mask, causal=causal)
    # Compute pytorch reference
    out_pt = attention_ref(q, k, v, query_padding_mask=attn_mask, key_padding_mask=attn_mask, causal=causal, 
                           upcast=False, reorder_ops=True)
    # Compute ours
    out = flash_attn_func(q, k, v, attn_mask, None, causal)
    # Compare results
    compare_results_fa(q, k, v, do, out, out_ref, out_pt)

@pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("local", [False, True]) # TODO: add support for window size?
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [32, 40, 59, 64, 80, 96, 111, 128]) # TODO: add support for head dim > 128, 160, 192, 224, 256])
@pytest.mark.parametrize("swap_seqlens, use_attention", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# @pytest.mark.parametrize("dropout_p", [0.0, 0.17]) TODO
@pytest.mark.parametrize("num_heads", [9])
@pytest.mark.parametrize("batch_size", [4])
def test_ragged(
    batch_size: int, 
    num_heads: int, 
    seqlen_q: int, 
    seqlen_k: int, 
    swap_seqlens: bool,
    head_dim: int, 
    causal: bool, 
    use_attention: bool, 
    dtype: torch.dtype,
) -> None: 
    if swap_seqlens:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    if use_attention:
        seqlen_q = seqlen_k
    _test_fwd_bwd(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, causal, use_attention, dtype)