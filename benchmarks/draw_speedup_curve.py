import sys
import os.path as osp
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))
from benchmarks.utils import measure_kernel_latency


COMPARE_TO = "Pytorch"
REPEATS = 10

BATCH_SIZES = [2**i for i in range(1)]
HIDDEN_DIM = 4096
NUM_HEADS = 32
DTYPE = torch.float16

POINTS_PER_BATCH = 10


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12))

    for batch_size in BATCH_SIZES:
        seqlens = torch.linspace(1, 2**14, steps=POINTS_PER_BATCH).int()
        measured_times = torch.zeros(size=(seqlens.size(0), 2, 2))
        oom_reached = False

        for i, seqlen in tqdm(enumerate(seqlens), total=POINTS_PER_BATCH):
            if oom_reached:
                break

            for j, kernel in enumerate([COMPARE_TO, "Liger"]):
                # Time kernel
                time_forward_and_backward = measure_kernel_latency(
                    kernel=kernel, 
                    repeats=REPEATS, 
                    batch_size=batch_size, 
                    num_heads=NUM_HEADS, 
                    seqlen=seqlen.item(), 
                    head_dim=HIDDEN_DIM // NUM_HEADS, 
                    causal=False,
                    dtype=DTYPE,
                )
                if time_forward_and_backward is None: # oom
                    seqlens = seqlens[:i]
                    measured_times = measured_times[:i]
                    oom_reached = True
                    break
                else:
                    measured_times[i, j] = torch.tensor(time_forward_and_backward)
        
        axs[0].set_title("Forward pass")
        axs[0].plot(seqlens, measured_times[:, 0, 0] / measured_times[:, 1, 0], label=str(batch_size))
        axs[1].set_title("Backward pass")
        axs[1].plot(seqlens, measured_times[:, 0, 1] / measured_times[:, 1, 1], label=str(batch_size))
        axs[2].set_title("Combined")
        axs[2].plot(seqlens, measured_times.sum(-1)[:, 0] / measured_times.sum(-1)[:, 1], label=str(batch_size))
        for i in range(3):
            axs[i].set_ylabel("Speedup")
            axs[i].axhline(1, color="red", linestyle="--")
        axs[2].set_xlabel("Sequence length")

    plt.plot()
    fig.tight_layout()
    fig.savefig(f"__tmp__.png")
    plt.close()
    