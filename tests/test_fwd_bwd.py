import pytest
import torch
import random

from tests.core import _test_core_fn

# TODO: add support for window size?
# TODO: add support for head dim > 128, 160, 192, 224, 256])

SKIP_PROB = 0.8


@pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dropout_p", [0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [32, 40, 59, 64, 80, 96, 111, 128])
@pytest.mark.parametrize(
    "swap_seqlens, use_attention, use_bias",
    [(False, False, False), (False, False, True), (False, True, False), (True, False, False), (True, False, True)],
)
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
@pytest.mark.parametrize("nheads_q, nheads_kv", [(8, 2), (9, 9)])
@pytest.mark.parametrize("batch_size", [4])
def test_fwd_bwd(
    batch_size: int,
    nheads_q: int,
    nheads_kv: int,
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
    if random.random() < SKIP_PROB:
        return None
    if swap_seqlens:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    if use_attention:
        seqlen_q = seqlen_k
    _test_core_fn(
        batch_size=batch_size,
        nheads_q=nheads_q,
        nheads_kv=nheads_kv,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        causal=causal,
        dropout_p=dropout_p,
        use_attention=use_attention,
        use_bias=use_bias,
        dtype=dtype,
        FORWARD_ONLY=False,
        RETURN=False,
    )
