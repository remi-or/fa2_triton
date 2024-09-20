from typing import Optional, Tuple
from torch import Tensor

import torch
import os
import matplotlib.pyplot as plt
import sys

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from src.wrapper import flash_attn_func
from tests.utils import generate_test_data, start_and_end, generate_attention_mask, compare_results_fa, compare_tensors
from src.other_implemenations.reference_implementation import attention_ref
from tests.test_repeatability import _test_repeatability

PLOT_HEAD_INDEX = None


batch_size = 1
num_heads = 9

seqlen_q = 1
seqlen_k = 239
swap_seqlens = True
use_attention = False

head_dim = 128
causal = False
dtype = torch.float16

forward_only = False



if __name__ == "__main__":
    # os.environ["TRITON_INTERPRET"] = "1"

    if swap_seqlens:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    if use_attention:
        seqlen_q = seqlen_k

    # Prepare data
    q, k, v, do = generate_test_data(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
    q_copy, k_copy, v_copy = q.clone(), k.clone(), v.clone()
    attn_mask = generate_attention_mask(q, True) if use_attention else None

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
        
        # Save a glimpse of the results
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(out.flatten(start_dim=1, end_dim=2)[-1].numpy(force=True))
        axs[1].imshow(out_pt.flatten(start_dim=1, end_dim=2)[-1].numpy(force=True))
        axs[2].imshow(out.flatten(start_dim=1, end_dim=2).sub(out_pt.flatten(start_dim=1, end_dim=2)).abs()[-1].numpy(force=True))
        fig.savefig("__tmp__.png")

        compare_results_fa(q, k, v, None, out, out_ref, out_pt)

        # Compare results
        compare_tensors(out, out_ref)
        exit(0)

    # Retrieve gradients
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), do, retain_graph=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), do, retain_graph=True)

    # Concatenate them along the number of heads
    if PLOT_HEAD_INDEX is None:
        dq, dq_pt, dq_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dq, dq_pt, dq_ref)]
        dk, dk_pt, dk_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dk, dk_pt, dk_ref)]
        dv, dv_pt, dv_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dv, dv_pt, dv_ref)]

    assert (q == q_copy).all()
    assert (k == k_copy).all()
    assert (v == v_copy).all()

    # Display part of the results
    print("Ours:", start_and_end(dk, 3))
    print("Ref:", start_and_end(dk_ref, 3))
    print("Pt:", start_and_end(dk_pt, 3))

    # Save a glimpse of the results
    fig, axs = plt.subplots(3, 3)
    b = 0
    for i, dxs in enumerate([(dq, dq_pt, dq_ref), (dk, dk_pt, dk_ref), (dv, dv_pt, dv_ref)]):
        if PLOT_HEAD_INDEX is None:
            axs[i, 0].imshow(dxs[0][b].float().numpy(force=True))
            axs[i, 1].imshow(dxs[1][b].float().numpy(force=True))
            axs[i, 2].imshow(dxs[0].sub(dxs[1]).abs()[b].float().numpy(force=True))
        else:
            axs[i, 0].imshow(dxs[0][b, :, PLOT_HEAD_INDEX].float().numpy(force=True))
            axs[i, 1].imshow(dxs[1][b, :, PLOT_HEAD_INDEX].float().numpy(force=True))
            axs[i, 2].imshow(dxs[0].sub(dxs[1]).abs()[b, :, PLOT_HEAD_INDEX].float().numpy(force=True))
        # axs[i, 2].imshow(dxs[2][-1].numpy(force=True))
    fig.savefig("__tmp__.png")

    # _test_repeatability(
    #     repeats=10, 
    #     batch_size=batch_size, 
    #     num_heads=num_heads, 
    #     seqlen_q=seqlen_q, 
    #     seqlen_k=seqlen_k, 
    #     head_dim=head_dim, 
    #     attention=use_attention, 
    #     causal=causal, 
    #     dtype=dtype,
    # )

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


# Precision issue with ################

# batch_size = 1
# num_heads = 9

# seqlen_q = 1
# seqlen_k = 239
# swap_seqlens = True
# use_attention = False

# head_dim = 59
# causal = False
# dtype = torch.float16

# [4-9-1-239-True-False-40-False-dtype0] FAILED                                                                                              [  5%]
# [4-9-1-239-True-False-32-False-dtype0] FAILED                                                                                              [  5%]

#######################################