from typing import Optional, Tuple
from torch import Tensor

import torch
import pytest
import os
import math
import matplotlib.pyplot as plt
import sys

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from src.kernel_wrapper import flash_attn_func
from src.kernel_callers import DEBUG_DELTA
from tests.utils import generate_test_data, generate_attention_mask, compare_tensors
from other_implemenations.reference_implementation import attention_ref

def check_backward_delta(
    batch_size: int, 
    num_heads: int, 
    seqlen_qk: int, 
    head_dim: int, 
    causal: bool, 
    dtype: torch.dtype = torch.float16,
    is_pytest: bool = False,
) -> Tuple[Tensor, Tensor]:  
    DEBUG_DELTA[0] = True
    softmax_scale = 1 / math.sqrt(head_dim)
    Q, K, V, dO = generate_test_data(batch_size, num_heads, seqlen_qk, seqlen_qk, head_dim, dtype)
    attention_mask = generate_attention_mask(Q)
    # Compute reference gradients
    ref_O = attention_ref(Q, K, V, query_padding_mask=attention_mask, key_padding_mask=attention_mask, causal=causal)
    ref_delta = (ref_O * dO).sum(-1)
    # Compute our O
    our_O: Tensor = flash_attn_func(Q, K, V, attention_mask, None, causal, softmax_scale) # arg[3] is bias, unused rn
    our_O.backward(dO)
    our_delta = Q.grad.clone()[:, :, :, 0]
    # Compare
    compare_tensors(ref_delta, our_delta, verbose=(not is_pytest), rtol=2, atol=1e-3) # TODO: lower rtol
    return ref_delta, our_delta

@pytest.mark.parametrize("batch_size, num_heads", [(1, 1), (2, 7), (3, 16), (4, 128)])
@pytest.mark.parametrize("seqlen_qk", [16, 65, 128, 150])
# @pytest.mark.parametrize("seqlen_k", [16, 65, 128, 150])
@pytest.mark.parametrize("head_dim", [12, 16, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_delta(    
    batch_size: int, 
    num_heads: int, 
    seqlen_qk: int, 
    head_dim: int, 
    causal: int, 
    dtype=torch.float16,
) -> None:
    check_backward_delta(batch_size, num_heads, seqlen_qk, head_dim, causal, dtype, is_pytest=True)


if __name__ == "__main__":
    # Delta: [B, S, H]
    ref, our = check_backward_delta(
        batch_size=16,
        num_heads=64,
        seqlen_qk=32,
        head_dim=12,
        causal=False,
    )

    ref = ref[0]
    our = our[0]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ref.numpy(force=True))
    axs[1].imshow(our.numpy(force=True))
    fig.savefig("__tmp__.png")