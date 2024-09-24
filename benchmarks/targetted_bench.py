import os
from typing import List, Tuple

import torch
from torch import Tensor

from benchmarks.utils import FA2TestCaller


# Params
BATCH_SIZE = 4
SEQLEN = 4096
NUM_HEADS = 32
HEAD_DIM = 4096 // NUM_HEADS

CAUSAL = False
DTYPE = torch.float16

FORWARD_ONLY = True


# Functions
def compute_masked_checksums(*tensors: Tuple[Tensor, ...]) -> List[float]:
    torch.manual_seed(1996)
    masked_checksums = []
    for t in tensors:
        mask = torch.rand_like(t).less_equal(0.1)
        masked_checksums.append(t[mask].sum().item())
    return masked_checksums


# Script
if __name__ == "__main__":
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    test_caller = FA2TestCaller(BATCH_SIZE, NUM_HEADS, SEQLEN, SEQLEN, 4096, False, False, torch.float16, False)

    # Loop over benched kernels
    for kernel in FA2TestCaller.available_kernels:
        test_caller.kernel = kernel

        # Benchmark he requested implementation
        times = test_caller.bench()
        # Get the masked checksum
        out, dq, dk, dv = test_caller()
        masked_checksums = compute_masked_checksums(out) if FORWARD_ONLY else compute_masked_checksums(out, dq, dk, dv)

        print("-" * 80)
        print("Kernel:", kernel)
        print("Measured time (ms):", times)
        print("Masked checksum:", masked_checksums)
    print()
