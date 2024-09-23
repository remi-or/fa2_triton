import torch
from src.forward.caller import _flash_attn_forward
from src.backward.caller import _flash_attn_backward


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attention_mask=None, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, attention_mask=attention_mask, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, attention_mask, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, attention_mask, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
        dq, dk, dv = _flash_attn_backward(
            do,
            q,
            k,
            v,
            attention_mask,
            o,
            lse,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None, None


flash_attn_func = FlashAttnFunc.apply


class LogsumexpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attention_mask=None, bias=None, causal=False, softmax_scale=None):
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, attention_mask=attention_mask, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        return lse

lse_func = LogsumexpFunc.apply