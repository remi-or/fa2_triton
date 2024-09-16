import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils import measure_kernel_latency

REPEATS = 5
MODE = "fwd_bwd"

BATCH_SIZES = [2**i for i in range(6)]
HIDDEN_DIM = 4096
NUM_HEADS = 32
DTYPE = torch.bfloat16


POINTS_PER_BATCH = 10

if __name__ == "__main__":

    head_dim = HIDDEN_DIM // NUM_HEADS

    for batch_size in BATCH_SIZES:
        flops = {kernel: [] for kernel in ["Liger", "Flex", "Pytorch"]} # "Liger", "Flex", "Pytorch"
        plt.close()

        seqlens = torch.linspace(1, 2**14, steps=POINTS_PER_BATCH).div(128).ceil().mul(128).int().tolist()
        for seqlen in tqdm(seqlens):

            for kernel in flops.keys():
                t = measure_kernel_latency(
                    kernel=kernel, 
                    mode=MODE,
                    repeats=REPEATS, 
                    batch_size=batch_size, 
                    num_heads=NUM_HEADS, 
                    seqlen=seqlen, 
                    head_dim=head_dim, 
                    causal=False,
                    dtype=DTYPE,
                )
                if t is None: # probably oom
                    continue
                flops[kernel].append(t)
        
        for kernel, ts in flops.items():
            plt.plot(seqlens[:len(ts)], ts, label=kernel)
    
        plt.legend()
        plt.savefig(f"__tmp_{batch_size}__.png")
        