import torch
from torch import Tensor

from torch.nn.attention.flex_attention import flex_attention as _flex_attention_uncompiled
from torch.nn.attention.flex_attention import create_block_mask

torch._dynamo.config.cache_size_limit = 1000
_flex_attention_compiled = torch.compile(_flex_attention_uncompiled, dynamic=False)

def causal_mask_fn(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
) -> Tensor:
    if causal:
        block_mask = create_block_mask(causal_mask_fn, B=None, H=None, Q_LEN=q.size(2), KV_LEN=k.size(2))
        return _flex_attention_compiled(q, k, v, block_mask=block_mask)
    return _flex_attention_compiled(
        query=q,
        key=k,
        value=v,
    )