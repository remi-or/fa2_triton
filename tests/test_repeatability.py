"""Here, we test if the kernel when there is no attention mask and seqlen_q != seqlen_k."""

import sys
sys.path.append("/home/remi_ouazan/fa2")
import torch
import pytest
from torch import Tensor

from src.kernel_wrapper import flash_attn_func
from tests.utils import generate_test_data, generate_attention_mask


def _test_repeatability(
    repeats: int,
    batch_size: int, 
    num_heads: int, 
    seqlen_q: int, 
    seqlen_k: int, 
    head_dim: int, 
    attention: bool,
    causal: bool, 
    dtype: torch.dtype = torch.float16,
) -> None: 
    # Prepare data
    if attention:
        seqlen_k = seqlen_q
        q, k, v, do = generate_test_data(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
        attn_mask = generate_attention_mask(q)
    else:
        q, k, v, do = generate_test_data(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
        attn_mask = None
    # Run computations
    checksums = {key: [] for key in ["out", "dq", "dk", "dv"]}
    mask_q = torch.rand_like(q).less_equal(0.05)
    mask_k = torch.rand_like(k).less_equal(0.05)
    for _ in range(repeats):
        out: Tensor = flash_attn_func(q, k, v, attn_mask, None, causal)    
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
        checksums["out"].append(out[mask_q].sum().item())
        checksums["dq"].append(dq[mask_q].sum().item())
        checksums["dk"].append(dk[mask_k].sum().item())
        checksums["dv"].append(dv[mask_k].sum().item())
    # Check sums
    distinct_checksums = {key: len(set(cks)) for key, cks in checksums.items()}
    assert all([x == 1 for x in distinct_checksums.values()]), checksums

# @pytest.mark.parametrize("dtype", ([torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("use_attention", [False, True])
# @pytest.mark.parametrize("head_dim", [32]) # TODO: add support for head dim > 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("swap_seqlens", [False, True])
# @pytest.mark.parametrize(
#     "seqlen_q,seqlen_k", 
#     [(1, 239), (3, 799), (127, 512), (127, 513), (113, 203), (128, 217), (113, 211), (108, 256), (256, 512), (1023, 1024)],
# )
# @pytest.mark.parametrize("num_heads", [9])
# @pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("dtype", ([torch.float16]))
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("attention", [True])
@pytest.mark.parametrize("head_dim", [32]) 
@pytest.mark.parametrize("swap_seqlens", [False])
@pytest.mark.parametrize("seqlen_q", [20, 32, 64, 79, 100, 164, 200, 239, 300])
@pytest.mark.parametrize("seqlen_k", [16])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("batch_size", [3])
def test_repeatability(
    batch_size: int, 
    num_heads: int, 
    seqlen_q: int, 
    seqlen_k: int, 
    swap_seqlens: bool,
    head_dim: int, 
    attention: bool,
    causal: bool, 
    dtype: torch.dtype,
) -> None: 
    if swap_seqlens: 
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    _test_repeatability(
        repeats=10, 
        batch_size=batch_size, num_heads=num_heads, seqlen_q=seqlen_q, seqlen_k=seqlen_k, head_dim=head_dim, 
        attention=attention, causal=causal, dtype=dtype)

if __name__ == "__main__":
    repeats = 10

    batch_size = 3
    num_heads = 1

    seqlen_q = 100
    seqlen_k = 16

    head_dim = 32
    attention= True

    causal = False
    dtype = torch.float16

    _test_repeatability(
        repeats=repeats, 
        batch_size=batch_size, num_heads=num_heads, seqlen_q=seqlen_q, seqlen_k=seqlen_k, head_dim=head_dim, 
        attention=attention, causal=causal, dtype=dtype)

# headdim=40, seqlen=(128, 117)

# There seems to be some problem with Triton pipelining that makes results wrong for
# headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
# may have zero step, and pipelining with the bias matrix could screw it up.
# So we just exit early.

# There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
# Also wrong for headdim=64.

# in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,

# There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
# Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
# Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False