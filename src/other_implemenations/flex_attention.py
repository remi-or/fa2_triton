import torch
from torch import Tensor

from torch.nn.attention.flex_attention import flex_attention as _flex_attention_uncompiled

torch._dynamo.config.cache_size_limit = 1000
_flex_attention_compiled = torch.compile(_flex_attention_uncompiled, dynamic=False)

def flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
) -> Tensor:
    if causal:
        raise NotImplementedError()
    return _flex_attention_compiled(
        query=q,
        key=k,
        value=v,
    )


# # FlexAttention-related functions
# def causal_mask_fn(b, h, q_idx, kv_idx):
#     return q_idx >= kv_idx

# def flex_attn_func(q, k, v, attn_mask, attn_bias, causal):
#     if causal:
#         block_mask = create_block_mask(causal_mask_fn, B=None, H=None, Q_LEN=1024, KV_LEN=1024)
#         return flex_attention(q, k, v, block_mask=block_mask)
#     else:
#         return flex_attention(q, k, v)