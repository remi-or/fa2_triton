
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

import torch
import math
from einops import repeat

from src.wrapper import lse_func
from tests.utils import generate_test_data, generate_attention_mask, compare_tensors

batch_size = 1
num_heads = 3

seqlen_q = 256
seqlen_k = 108
swap_seqlens = False
use_attention = False

head_dim = 32
causal = True
dtype = torch.float16

forward_only = False


if __name__ == "__main__":
    
    # Prepare data
    q, k, v, _ = generate_test_data(batch_size, num_heads, seqlen_q, seqlen_k, head_dim, dtype)
    attn_mask = generate_attention_mask(q) if use_attention else None
    # Compute reference
    q, k, v = q.half(), k.half(), v.half()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d)).float()

    if causal:
        # Create causal mask
        causal_mask = torch.tril(torch.ones(size=(seqlen_q, seqlen_k))).bool().cuda()
        # Add the batch and head dimension
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Repeat for batch
        causal_mask = causal_mask.repeat((batch_size, 1, 1, 1))
        # Apply attention mask on top
        if use_attention:
            for i in range(batch_size):
                causal_mask[i, :, :, attn_mask[i].logical_not()] = 0
        # Compute lse
        p = torch.where(causal_mask, scores, float("-inf")).softmax(dim=-1)
        log_p = torch.where(causal_mask, p.log(), 0)
        lse_ref = (torch.where(causal_mask, scores, 0) - log_p).sum(-1) / causal_mask.sum(-1)
    else:
        p = scores.softmax(dim=-1)
        log_p = p.log()
        lse_ref = (scores - log_p).mean(-1)
    lse_ref *= 1.44269504089

    # Compute ours
    lse = lse_func(q, k, v, attn_mask, None, causal)
    lse = lse[:, :, :seqlen_q]
    # Compare results
    compare_tensors(lse_ref, lse, verbose=True)