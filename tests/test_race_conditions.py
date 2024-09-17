import torch
from tests.test_fwd_only import _test_fwd_only

FOUND_RACE_CONDITION_CFGS = [
    (1, 1, 128, 128, False, False, 17, False, torch.float16),
    (4, 9, 127, 512, False, True, 40, False, torch.float16),
]

if __name__ == "__main__":

    for cfg in FOUND_RACE_CONDITION_CFGS:
        batch_size, num_heads, seqlen_q, seqlen_k, swap_seqlens, use_attention, head_dim, causal, dtype = cfg
        _test_fwd_only(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, causal, use_attention, dtype)