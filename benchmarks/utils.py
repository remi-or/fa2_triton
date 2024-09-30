from typing import Optional, Tuple

import torch
from torch import Tensor
import triton

from src.reference_implementation import flash_attn_reference
from src.wrapper import flash_attn_func
from tests.utils import generate_attention_mask, generate_test_data

try:
    from src.other_implementations.flex_attention import (
        _flex_attention_compiled,
        causal_mask_fn,
        create_block_mask,
    )
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False


class FA2TestCaller:

    available_kernels = ["Liger", "Pytorch"] + (["Flex"] if FLEX_AVAILABLE else [])

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seqlen_q: int,
        seqlen_k: int,
        hidden_dim: int,
        use_attention: bool,
        causal: bool,
        dtype: torch.dtype,
        forward_only: bool = False,
    ) -> None:
        self.q, self.k, self.v, self.do = generate_test_data(
            batch_size=batch_size,
            num_heads=num_heads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            head_dim=hidden_dim // num_heads,
            dtype=dtype,
        )
        self.attn_mask = generate_attention_mask(self.q) if use_attention else None
        if FLEX_AVAILABLE and causal:
            self.block_mask = create_block_mask(causal_mask_fn, None, None, seqlen_q, seqlen_k)
        else:
            self.block_mask = None
        self.causal = causal
        self.forward_only = forward_only
        self._kernel = ""

    @property
    def kernel(self) -> str:
        if not self._kernel:
            raise RuntimeError("Tried to access the proprety kernel without first setting it.")
        return self._kernel

    @kernel.setter
    def kernel(self, value: str) -> None:
        if value not in self.available_kernels:
            raise ValueError(f"Cannot set the kernel to {value}. Available kernels are {self.available_kernels}")
        if (self._kernel == "Flex") != (value == "Flex"):
            self.q = self.q.transpose(1, 2).contiguous()
            self.k = self.k.transpose(1, 2).contiguous()
            self.v = self.v.transpose(1, 2).contiguous()
            self.do = self.do.transpose(1, 2).contiguous()
        self._kernel = value

    def __call__(self) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Forward pass
        if self.kernel == "Liger":
            out = flash_attn_func(self.q, self.k, self.v, self.attn_mask, self.causal)
        elif self.kernel == "Flex":
            out = _flex_attention_compiled(self.q, self.k, self.v, block_mask=self.block_mask)
        elif self.kernel == "Pytorch":
            out = flash_attn_reference(self.q, self.k, self.v, self.attn_mask, self.attn_mask, causal=self.causal)
        else:
            raise KeyError(f"Unknown self.kernel: {self.kernel}")
        # Maybe backward pass
        if not self.forward_only:
            dq, dk, dv = torch.autograd.grad(out, (self.q, self.k, self.v), self.do)
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
        else:
            dq, dk, dv = None, None, None
        return out, dq, dk, dv

    def bench(self, warmup: int = 100, rep: int = 1000) -> float:
        return triton.testing.do_bench(fn=self, warmup=warmup, rep=rep)
