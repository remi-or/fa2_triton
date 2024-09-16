"""Here, we test if the kernel when there is an attention mask and seqlen_q == seqlen_k."""

import sys
sys.path.append("/home/remi_ouazan/fa2") # HOTFIX
import torch
import pytest

from src.kernel_wrapper import flash_attn_func
from tests.utils import generate_test_data, generate_attention_mask, compare_results_fa
from other_implemenations.reference_implementation import attention_ref


def _test_attn(
    batch_size: int, 
    num_heads: int, 
    seqlen: int, 
    head_dim: int, 
    causal: bool, 
    dtype: torch.dtype = torch.float16,
) -> None: 
    # Prepare data
    q, k, v, do = generate_test_data(batch_size, num_heads, seqlen, seqlen, head_dim, dtype)
    attn_mask = generate_attention_mask(q)
    # Compute reference
    out_ref = attention_ref(q, k, v, query_padding_mask=attn_mask, key_padding_mask=attn_mask, causal=causal)
    # Compute pytorch reference
    out_pt = attention_ref(
        q, 
        k, 
        v, 
        query_padding_mask=attn_mask, 
        key_padding_mask=attn_mask, 
        causal=causal, 
        upcast=False, 
        reorder_ops=True,
    )
    # Compute ours
    out = flash_attn_func(q, k, v, attn_mask, None, causal)
    # Compare results
    compare_results_fa(q, k, v, do, out, out_ref, out_pt)

@pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("local", [False, True]) # TODO: add support for window size?
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [32, 40, 59, 64, 80, 96, 111, 128]) # TODO: add support for head dim > 128, 160, 192, 224, 256])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 384, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize("dropout_p", [0.0, 0.17]) # TODO: add dropout support
@pytest.mark.parametrize("num_heads", [9])
@pytest.mark.parametrize("batch_size", [4])
def test_basic(
    batch_size: int, 
    num_heads: int, 
    seqlen: int, 
    head_dim: int, 
    causal: bool, 
    dtype: torch.dtype,
) -> None: 
    _test_attn(batch_size, num_heads, seqlen, head_dim, causal, dtype)