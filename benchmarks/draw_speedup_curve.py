import sys
import os.path as osp
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List
from torch import Tensor

sys.path.append(osp.dirname(osp.dirname(__file__)))
from benchmarks.utils import measure_kernel_latency


COMPARE_TO = "Pytorch"
REPEATS = 20

BATCH_SIZES = [2**i for i in range(6)]
HIDDEN_DIM = 4096
NUM_HEADS = 32
DTYPE = torch.float16

def retrieve_speedup_curve_data(
    kernels: List[str],
    batch_size: int,
    seqlens: Tensor,
    causal: bool,
    use_attention: bool,
) -> Tensor:
    # Setup
    measured_times = torch.zeros(size=(seqlens.size(0), len(kernels), 2))
    # Loop on seqlen
    for i, seqlen in tqdm(enumerate(seqlens), total=14):
        # Loop on kernels
        for j, kernel in enumerate(kernels):
            # Time kernel
            time_forward_and_backward = measure_kernel_latency(
                kernel=kernel, 
                repeats=REPEATS, 
                batch_size=batch_size, 
                num_heads=NUM_HEADS, 
                seqlen=seqlen.item(), 
                head_dim=HIDDEN_DIM // NUM_HEADS, 
                causal=causal,
                use_attention=use_attention,
                dtype=DTYPE,
            )
            # Accumulate or stop because of OOM
            if time_forward_and_backward is None:
                return seqlens[:i], measured_times[:i]
            else:
                measured_times[i, j] = torch.tensor(time_forward_and_backward)
    return seqlens, measured_times


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12))   
    axs[0].set_title("Forward pass")
    axs[1].set_title("Backward pass")
    axs[2].set_title("Combined")
    for i in range(3):
        axs[i].set_ylabel("Speedup")
        axs[i].axhline(1, color="red", linestyle="--")
    axs[2].set_xlabel("Sequence length")

    for batch_size in BATCH_SIZES:
        seqlens, measured_times = retrieve_speedup_curve_data(
            kernels=["Pytorch", "Liger"],
            batch_size=batch_size, 
            seqlens=torch.arange(1, 15).exp2().int(),
            causal=False,
            use_attention=False,
        )
        
        axs[0].plot(seqlens, measured_times[:, 0, 0] / measured_times[:, 1, 0], label=f"B = {batch_size}")
        axs[1].plot(seqlens, measured_times[:, 0, 1] / measured_times[:, 1, 1], label=f"B = {batch_size}")
        axs[2].plot(seqlens, measured_times.sum(-1)[:, 0] / measured_times.sum(-1)[:, 1], label=f"B = {batch_size}")
        fig.tight_layout()
        fig.savefig(f"__tmp_speedup.png")
    
    for i in range(3):
        axs[i].legend()
    fig.tight_layout()
    fig.savefig(f"__tmp_speedup.png")
    plt.close()
    