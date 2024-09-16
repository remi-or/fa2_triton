import torch

def attention_pack(
    x: torch.Tensor, # [batch_size, seqlen, num_heads, head_dim]
    attention_mask: torch.Tensor, # [batch_size, seqlen]
) -> torch.Tensor:
    to_pack = [] 
    for i, attn_mask in enumerate(attention_mask):
        seqlen = attn_mask.sum().int().item()
        kept = x[i, :seqlen] # [seqlen, num_heads, head_dim]
        to_pack.append(kept)
    return torch.concatenate(to_pack, dim=0).unsqueeze(0)

def attention_unpack(
    x: torch.Tensor, # [1, sum_seqlens, num_heads, head_dim]
    cum_seqlens: torch.Tensor, # [0, seqlen_1, seqlen_1+seqlen_2, ...]
    batch_size: int,
    goal_seqlen: int,
) -> torch.Tensor:
    unpacked = torch.zeros(size=(batch_size, goal_seqlen, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(cum_seqlens.size(0)-1):
        seq_start = cum_seqlens[i]
        seq_end = cum_seqlens[i+1]
        unpacked[i, :seq_end-seq_start] = x[0, seq_start:seq_end]
    return unpacked