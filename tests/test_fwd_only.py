import pytest
import torch

from tests.core import _test_core_fn


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
def test_fwd_only(
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
    _test_core_fn(
        batch_size=batch_size,
        nheads_q=num_heads,
        nheads_kv=num_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        causal=causal,
        dropout_p=dropout_p,
        use_attention=use_attention,
        use_bias=use_bias,
        dtype=dtype,
        FORWARD_ONLY=True,
        RETURN=False,
    )
