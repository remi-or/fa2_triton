import torch
from einops import repeat
import math

from src.forward.caller import _flash_attn_forward
from tests.utils import generate_test_data, generate_attention_mask, compare_tensors, generate_dropout_seed_and_mask

batch_size = 1
nheads_q = 1
nheads_kv = 1
head_dim = 32

seqlen_q = 256
seqlen_k = 256
swap_seqlens = False

use_attention = False
use_bias = False
dropout_p = 0.17
causal = True

dtype = torch.float16


if __name__ == "__main__":
    raise NotImplementedError()

    # Prepare data
    q, k, v, do = generate_test_data(batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k, head_dim, dtype)
    attn_mask = generate_attention_mask(q) if use_attention else None
    attn_bias = torch.rand(size=(1, 1, seqlen_q, seqlen_k), dtype=dtype, device="cuda") if use_bias else None
    dropout_seed, dropout_mask = generate_dropout_seed_and_mask(dropout_p, q, k, attn_mask)

    # Compute the LSE
    lse = _flash_attn_forward(
        q=q,
        k=k,
        v=v,
        attention_mask=attn_mask,
        bias=attn_bias,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=None,
        dropout_seed=dropout_seed,
    )[1]

    # Compute the reference LSE
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
