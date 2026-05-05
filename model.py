import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from attention import flash_attention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq, dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (RoPE)"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to queries and keys.
    xq: (B, T, H, D)
    xk: (B, T, KV_H, D)
    freqs_cis: (T, D/2)
    """
    # Reshape xq and xk to view the head_dim as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to match xq_ and xk_ shape: (1, T, 1, D/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Complex multiply
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        
        # KV cache tensors will be registered during inference if needed
        self.cache_k = None
        self.cache_v = None
        
    def forward(self, x, freqs_cis, mask=None, start_pos=None, use_cache=False):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # KV Cache logic
        if use_cache:
            if start_pos == 0 or self.cache_k is None:
                # Initialize or reset cache
                self.cache_k = torch.zeros((B, self.config_ctx_len, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros((B, self.config_ctx_len, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)
            
            self.cache_k[:, start_pos : start_pos + T] = k
            self.cache_v[:, start_pos : start_pos + T] = v
            
            k = self.cache_k[:, : start_pos + T]
            v = self.cache_v[:, : start_pos + T]
        
        # Repeat k, v heads to match q heads
        if self.n_rep > 1:
            k = torch.repeat_interleave(k, self.n_rep, dim=2)
            v = torch.repeat_interleave(v, self.n_rep, dim=2)
            
        # Flash attention expects (B, T, H, D)
        # Note: causal mask is only needed if T > 1
        out = flash_attention(q, k, v, causal=(T > 1))
        
        # Output projection
        out = out.reshape(B, T, -1)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.ffn_hidden_dim)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, freqs_cis, mask=None, start_pos=None, use_cache=False):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, start_pos, use_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Add ctx_len to attention blocks for cache sizing
        for layer in self.layers:
            layer.attention.config_ctx_len = config.ctx_len

        self.norm = RMSNorm(config.d_model)
        # Weight tying
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(config.d_model // config.n_heads, config.ctx_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # Init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, start_pos=None, use_cache=False, use_checkpointing=False):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Get frequencies for current sequence length starting at start_pos
        if start_pos is not None:
            freqs_cis = self.freqs_cis[start_pos : start_pos + T].to(h.device)
        else:
            freqs_cis = self.freqs_cis[:T].to(h.device)
        
        for layer in self.layers:
            if use_checkpointing:
                h = torch.utils.checkpoint.checkpoint(layer, h, freqs_cis, None, start_pos, use_cache, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, start_pos=start_pos, use_cache=use_cache)
            
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
