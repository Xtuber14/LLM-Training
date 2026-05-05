import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for PyTorch 2.0+ native SDPA first as it is most compatible
if hasattr(F, "scaled_dot_product_attention"):
    USE_FLASH = "sdpa"
else:
    # Try importing xformers
    try:
        from xformers.ops import memory_efficient_attention, LowerTriangularMask
        USE_FLASH = "xformers"
    except ImportError:
        try:
            from flash_attn import flash_attn_func
            USE_FLASH = "flash_attn"
        except ImportError:
            USE_FLASH = "chunked"

def get_attention_backend():
    return USE_FLASH

def apply_chunked_attention(q, k, v, mask=None, chunk_size=512):
    """
    Fallback attention implementation that bounds VRAM usage.
    Processes Q in chunks of chunk_size to compute attention.
    Expected shapes:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_heads, seq_len, head_dim)
        v: (batch, n_heads, seq_len, head_dim)
    """
    B, H, T, D = q.size()
    out = torch.zeros_like(q)
    
    scale = 1.0 / math.sqrt(D)
    
    # Precompute indices for mask
    k_idx = torch.arange(T, device=k.device).unsqueeze(0)        # (1, T)
    
    # Process queries in chunks to avoid O(T^2) memory footprint
    for i in range(0, T, chunk_size):
        end_i = min(i + chunk_size, T)
        q_chunk = q[:, :, i:end_i, :]  # (B, H, chunk, D)
        
        # Compute scores for this chunk vs all keys
        # scores: (B, H, chunk, T)
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        q_idx = torch.arange(i, end_i, device=q.device).unsqueeze(1) # (chunk, 1)
        causal_mask = k_idx > q_idx                                  # (chunk, T)
        
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores + mask[:, :, i:end_i, :]
            
        attn = F.softmax(scores, dim=-1)
        # Handle nan for padded sequences
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Multiply by V
        out_chunk = torch.matmul(attn, v)  # (B, H, chunk, D)
        out[:, :, i:end_i, :] = out_chunk
        
    return out

def flash_attention(q, k, v, causal=True):
    """
    Dispatcher for attention implementations.
    Expects inputs in shape: (batch, seq_len, n_heads, head_dim)
    """
    if USE_FLASH == "xformers":
        return memory_efficient_attention(
            q, k, v, 
            attn_bias=LowerTriangularMask() if causal else None
        )
        
    elif USE_FLASH == "flash_attn":
        return flash_attn_func(q, k, v, causal=causal)
    
    elif USE_FLASH == "sdpa":
        # Native PyTorch SDPA expects (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out.transpose(1, 2).contiguous()
        
    else:
        # Fallback expects (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = apply_chunked_attention(q, k, v)
        
        # Back to (B, T, H, D)
        return out.transpose(1, 2).contiguous()
