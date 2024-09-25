import pytest
import torch

from src.reference_implementation import attention_ref
from src.wrapper import flash_attn_func
from tests.utils import compare_results_fa, generate_attention_mask, generate_test_data, generate_dropout_seed_and_mask


def _test_fwd_only(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    causal: bool,
    dropout_p: float,
    use_attention: bool,
    use_bias: bool,
    dtype: torch.dtype = torch.float16,
) -> None:
    # Prepare data
    q, k, v, _ = generate_test_data(batch_size, num_heads, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
    attn_mask = generate_attention_mask(q) if use_attention else None
    attn_bias = torch.rand(size=(1, 1, seqlen_q, seqlen_k), dtype=dtype, device="cuda") if use_bias else None
    dropout_seed, dropout_mask = generate_dropout_seed_and_mask(dropout_p, q, k, attn_mask)
    # Compute reference
    out_ref = attention_ref(
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
    out_pt = attention_ref(
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
    compare_results_fa(q, k, v, None, out, out_ref, out_pt)


@pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("local", [False, True]) # TODO: add support for window size?
@pytest.mark.parametrize("dropout_p", [0, 0.1])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [32, 40, 59, 64, 80, 96, 111, 128])  # TODO: add support for head dim > 128, 160, 192, 224, 256])
@pytest.mark.parametrize("swap_seqlens, use_attention, use_bias", [(False, False, True), (True, False, True)])
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
    dropout_p: float,
    use_attention: bool,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    if swap_seqlens:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    if use_attention:
        seqlen_q = seqlen_k
    _test_fwd_only(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, causal, dropout_p, use_attention, use_bias, dtype)
