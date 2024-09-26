import os

import matplotlib.pyplot as plt
import torch

from tests.utils import start_and_end, compare_results_fa, compare_tensors
from tests.core import _test_core_fn

PLOT_HEAD_INDEX = None

batch_size = 1
num_heads = 1

seqlen_q = 32
seqlen_k = 32
swap_seqlens = False
use_attention = False

head_dim = 32
causal = False
dtype = torch.float16

forward_only = False


if __name__ == "__main__":
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    if swap_seqlens:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    if use_attention:
        seqlen_q = seqlen_k

    q, k, v, out, out_pt, out_ref, do = _test_core_fn(
        batch_size=batch_size,
        nheads_q=num_heads,
        nheads_kv=num_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        causal=causal,
        dropout_p=0.99,
        use_attention=use_attention,
        use_bias=False,
        dtype=dtype,
        FORWARD_ONLY=True,
        RETURN=True,
    )

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

    print(dq / dq_ref)

    # Concatenate them along the number of heads
    if PLOT_HEAD_INDEX is None:
        dq, dq_pt, dq_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dq, dq_pt, dq_ref)]
        dk, dk_pt, dk_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dk, dk_pt, dk_ref)]
        dv, dv_pt, dv_ref = [x.flatten(start_dim=1, end_dim=2) for x in (dv, dv_pt, dv_ref)]

    # Display part of the results
    print("Ours:", start_and_end(dq, 3))
    print("Ref:", start_and_end(dq_ref, 3))
    print("Pt:", start_and_end(dq_pt, 3))

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

# [4-9-1-239-True-False-40-False-dtype0] - ArithmeticError: Gradient of V. sum_dv_error = 0.005859375
# [4-9-1-239-True-False-64-False-dtype0] - ArithmeticError: Gradient of V. sum_dv_error = 0.001953125
# [4-9-1-239-True-False-96-False-dtype0] - ArithmeticError: Gradient of V. sum_dv_error = 0.0009765625
# [4-9-127-513-False-True-32-True-dtype0] - AssertionError: Gradient of Q
# [4-9-108-256-True-False-32-True-dtype0] - AssertionError: Gradient of K
# [4-9-108-256-True-False-40-True-dtype0] - AssertionError: Gradient of K
# [4-9-108-256-True-False-80-True-dtype0] - AssertionError: Gradient of K
