from typing import Optional, Tuple
from torch import Tensor

import torch
import os
import matplotlib.pyplot as plt
import sys

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from src.kernel_wrapper import flash_attn_func
from tests.utils import generate_test_data, start_and_end, generate_attention_mask, compare_results_fa, compare_tensors
from src.other_implemenations.reference_implementation import attention_ref

batch_size = 4
num_heads = 9

seqlen_q = 97
seqlen_k = 97

head_dim = 64
attention = False

causal = False
dtype = torch.float16

forward_only = False


if __name__ == "__main__":
    if attention:
        assert seqlen_q == seqlen_k

    # Prepare data
    q, k, v, do = generate_test_data(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
    attn_mask = generate_attention_mask(q, True) if attention else None

    # Compute reference
    out_ref = attention_ref(q, k, v, query_padding_mask=attn_mask, key_padding_mask=attn_mask, causal=causal)
    # Compute pytorch reference
    out_pt = attention_ref(
        q, k, v, query_padding_mask=attn_mask, key_padding_mask=attn_mask, causal=causal, upcast=False, reorder_ops=True
    )
    # Compute ours
    out = flash_attn_func(q, k, v, attn_mask, None, causal)

    if forward_only:
        # Display part of the results
        print("Ours:", start_and_end(out, 3))
        print("Ref:", start_and_end(out_ref, 3))
        print("Pt:", start_and_end(out_pt, 3))

        out, out_pt, out_ref = [x.flatten(start_dim=1, end_dim=2) for x in (out, out_pt, out_ref)]

        # Save a glimpse of the results
        fig, axs = plt.subplots(1, 3)
        for i, x in enumerate([out, out_pt, out_ref]):
            axs[i].imshow(x[-1].numpy(force=True))
            axs[i].imshow(x[-1].numpy(force=True))
            axs[i].imshow(x[-1].numpy(force=True))
        fig.savefig("__tmp__.png")

        # Compare results
        compare_tensors(out, out_ref)
        exit(0)

    # Retrieve gradients
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), do, retain_graph=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), do, retain_graph=True)

    # Concatenate them along the number of heads
    dq, dq_pt, dq_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dq, dq_pt, dq_ref)]
    dk, dk_pt, dk_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dk, dk_pt, dk_ref)]
    dv, dv_pt, dv_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dv, dv_pt, dv_ref)]

    # Display part of the results
    print("Ours:", start_and_end(dq, 3))
    print("Ref:", start_and_end(dq_ref, 3))
    print("Pt:", start_and_end(dq_pt, 3))

    # Save a glimpse of the results
    fig, axs = plt.subplots(3, 3)
    for i, dxs in enumerate([(dq, dq_pt, dq_ref), (dk, dk_pt, dk_ref), (dv, dv_pt, dv_ref)]):
        axs[i, 0].imshow(dxs[0][-1].numpy(force=True))
        axs[i, 1].imshow(dxs[1][-1].numpy(force=True))
        axs[i, 2].imshow(dxs[2][-1].numpy(force=True))
    fig.savefig("__tmp__.png")

    # Compare results
    compare_results_fa(q, k, v, do, out, out_ref, out_pt)
    
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