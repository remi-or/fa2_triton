from typing import Optional

import torch
from torch import Tensor

from src.backward.caller import _flash_attn_backward
from src.forward.caller import _flash_attn_forward


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal: bool = False,
        softmax_scale: Optional[Tensor] = None,
    ):
        """
        Compute the forward pass of the FlashAttention function.
        Args:
            - ctx (): the autograd.Function context
            - q (Tensor): the query projection tensor, of shape [batch_size, seqlen_q, num_heads, head_dim]
            - k (Tensor): the key projection tensor, of shape [batch_size, seqlen_k, num_heads, head_dim]
            - v (Tensor): the values projection tensor, of shape [batch_size, seqlen_k, num_heads, head_dim]
            - attention_mask (Optional[Tensor]): an optional attention mask of shape [batch_size, seqlen_q].
                Forces seqlen_q == seqlen_k.
            - causal (bool): a boolean to indicate whether or not to use causal attention
            - softmax_scale (Optional[float]): an optional float to scale the pre-softmax attention scores. Defaults
                to 1 / sqrt(head_dim)
        Return:
            the attention output tensor
        """
        # Make sure that the last dimension is contiguous
        q = q if q.stride(-1) == 1 else q.contiguous()
        k = k if k.stride(-1) == 1 else k.contiguous()
        v = v if v.stride(-1) == 1 else v.contiguous()
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, attention_mask=attention_mask, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, attention_mask, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Compute the backward pass of the FlashAttention function.
        Args:
            - ctx (): the autograd.Function context
            - do (Tensor): the gradient of the output tensor, of shape [batch_size, seqlen_q, num_heads, head_dim]
        Return:
            three tensors, the gradients of q, k and v respectively (check forward for shape info)
        """
        q, k, v, attention_mask, o, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(
            do, q, k, v, attention_mask, o, lse, causal=ctx.causal, softmax_scale=ctx.softmax_scale
        )
        return dq, dk, dv, None, None, None, None


flash_attn_func = FlashAttnFunc.apply


class LogsumexpFunc(torch.autograd.Function):
    """A function to retrieve the LogSumExp of the forward pass of FA2. Mainly for debug purposes."""

    @staticmethod
    def forward(ctx, q, k, v, attention_mask=None, bias=None, causal=False, softmax_scale=None):
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, attention_mask=attention_mask, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        return lse


lse_func = LogsumexpFunc.apply
