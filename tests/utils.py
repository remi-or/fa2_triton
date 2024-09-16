import torch
from torch import Tensor
from typing import Tuple, Optional

def generate_test_data(
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int = 0,
) -> Tuple[Tensor, ...]:
    """Generate the data necessary for a test of the FlashAttention2 algorithm: Q, K, V and dO."""
    torch.manual_seed(seed) 
    Q = (torch.empty((batch_size, seqlen_q, num_heads, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    K = (torch.empty((batch_size, seqlen_k, num_heads, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    V = (torch.empty((batch_size, seqlen_k, num_heads, head_dim), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    dO = torch.randn_like(Q)
    return Q, K, V, dO

def retrieve_and_wipe_gradients(
    Q: Tensor, K: Tensor, V: Tensor, O: Tensor, dO: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Retrieves and wipe the gradients from the queries, keys and values tensors."""
    O.backward(dO)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None
    return dQ, dK, dV

def generate_attention_mask(x: Tensor, verbose: bool = False) -> Tensor:
    """Generates a random attention mask for a tensor of shape [batch_size, seqlen, ...]"""
    attention_mask = torch.ones(size=x.shape[:2], dtype=bool, device=x.device)
    # If seqlen == 1, the attention mask is full of ones
    if x.size(1) == 1: 
        return attention_mask
    # Otherwise, choose a random padding per batch
    padding_per_batch = torch.randint(low=0, high=x.size(1)-1, size=(x.size(0), )).tolist()
    for i, padding in enumerate(padding_per_batch):
        attention_mask[i, -padding:] = 0
    if verbose:
        print("Paddings:", padding_per_batch)
    return attention_mask

def start_and_end(x: Tensor, num_elem: int) -> str:
    """"Returns a string with the (num_elem) first and last elements of the given tensor (x)."""
    if num_elem * 2 >= x.numel():
        return repr(x.flatten().tolist())
    start = x.flatten().tolist()[:num_elem]
    end = x.flatten().tolist()[-num_elem:]
    return repr(start) + " ... " + repr(end)


def compare_results_fa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    do: Optional[Tensor],
    out: Tensor,
    out_ref: Tensor,
    out_pt: Tensor,
    out_error_mul: int = 2,
    out_error_bias: float = 2e-5,
    grad_error_mul: int = 3, # 2 or 3 
    grad_error_bias: float = 1e-5, # 0 or 1e-5
) -> Tuple[Optional[Tensor], ...]:
    """This code is a slightly modified version of the one from the original FlashAttention repo: 
        https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py """
    
    # Display output-related diffs
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # if dropout_p > 0.0:
    #     print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
    #     print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")
    # Fail test if diffs are to high
    assert (out - out_ref).abs().max().item() <= out_error_mul * (out_pt - out_ref).abs().max().item() + out_error_bias, "Output"

    # Stop here if this is forward only mode
    forward_only_mode = (do is None)
    if forward_only_mode:
        return None, None, None

    dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), do)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), do)
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.

    # if dropout_p > 0.0:
        # assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        # if not alibi:
        #     assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    max_dq_error = (dq - dq_ref).abs().max().item()
    assert max_dq_error <= grad_error_mul * (dq_pt - dq_ref).abs().max().item() + grad_error_bias, "Gradient of Q"
    max_dk_error = (dk - dk_ref).abs().max().item()
    assert max_dk_error <= grad_error_mul * (dk_pt - dk_ref).abs().max().item() + grad_error_bias, "Gradient of K"
    max_dv_error = (dv - dv_ref).abs().max().item()
    assert max_dv_error <= grad_error_mul * (dv_pt - dv_ref).abs().max().item() + grad_error_bias, "Gradient of V"

    return dq, dk, dv


def compare_tensors(
    reference: Tensor, 
    ours: Tensor, 
    rtol: float = 1, 
    atol: float = 1e-4,
    verbose: bool = False,
) -> None:
    assert reference.shape == ours.shape, f"{reference.shape = } != {ours.shape = }"
    nonzero_reference = reference.masked_fill(reference.abs().less_equal(1e-4), 1)
    rtol_max_error = reference.sub(ours).div(nonzero_reference).max().item()
    atol_max_error = reference.sub(ours).abs().max().item()
    details = f"{rtol_max_error = } | {atol_max_error = }"
    # Display a bit of each tensor
    if verbose:
        print("Ref:", start_and_end(reference, 3))
        print("Our:", start_and_end(ours, 3))
        print(details)
    if not torch.allclose(reference, ours, rtol=rtol, atol=atol):
        if verbose:
            print("-"*30, "TESTED FAILED", "-"*30)
        else:
            raise ValueError(details)