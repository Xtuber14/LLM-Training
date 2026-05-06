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

def precompute_rope_frequencies(dim: int, end: int, theta: float = 10000.0):
    """Precompute RoPE sine/cosine tables in real space for better portability."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Backwards-compatible helper used by tests."""
    cos, sin = precompute_rope_frequencies(dim, end, theta)
    return torch.complex(cos, sin)

def _rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)

def apply_rotary_emb(xq, xk, freqs_cis=None, cos=None, sin=None):
    """
    Apply rotary embeddings to queries and keys.
    xq: (B, T, H, D)
    xk: (B, T, KV_H, D)
    freqs_cis: (T, D/2) complex tensor, kept for backwards compatibility
    cos/sin: (T, D/2) real-valued RoPE tables
    """
    if cos is None or sin is None:
        if freqs_cis is None:
            raise ValueError("Either freqs_cis or both cos/sin must be provided.")
        cos = freqs_cis.real
        sin = freqs_cis.imag

    cos = torch.repeat_interleave(cos.unsqueeze(0).unsqueeze(2), 2, dim=-1)
    sin = torch.repeat_interleave(sin.unsqueeze(0).unsqueeze(2), 2, dim=-1)
    cos = cos.to(device=xq.device, dtype=xq.dtype)
    sin = sin.to(device=xq.device, dtype=xq.dtype)

    xq_out = (xq * cos) + (_rotate_half(xq) * sin)
    xk_out = (xk * cos) + (_rotate_half(xk) * sin)
    return xq_out, xk_out

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
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        
        self.cache_k = None
        self.cache_v = None
        self.cache_seq_len = 0

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_seq_len = 0

    def _allocate_cache(self, batch_size, device, dtype):
        shape = (batch_size, self.config.ctx_len, self.n_kv_heads, self.head_dim)
        self.cache_k = torch.empty(shape, device=device, dtype=dtype)
        self.cache_v = torch.empty(shape, device=device, dtype=dtype)
        self.cache_seq_len = 0
        
    def forward(self, x, cos, sin, mask=None, start_pos=None, use_cache=False):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, cos=cos, sin=sin)
        
        if use_cache:
            if start_pos is None:
                start_pos = self.cache_seq_len
            if start_pos + T > self.config.ctx_len:
                raise ValueError(
                    f"KV cache overflow: requested position {start_pos + T}, "
                    f"but ctx_len={self.config.ctx_len}"
                )
            needs_reset = (
                self.cache_k is None
                or self.cache_v is None
                or self.cache_k.size(0) != B
                or self.cache_k.device != x.device
                or self.cache_k.dtype != x.dtype
                or start_pos == 0
            )
            if needs_reset:
                self._allocate_cache(B, x.device, x.dtype)

            end_pos = start_pos + T
            self.cache_k[:, start_pos:end_pos].copy_(k)
            self.cache_v[:, start_pos:end_pos].copy_(v)
            self.cache_seq_len = max(self.cache_seq_len, end_pos)

            k = self.cache_k[:, :self.cache_seq_len]
            v = self.cache_v[:, :self.cache_seq_len]

        causal = T > 1 and (not use_cache or start_pos == 0)
        out = flash_attention(q, k, v, causal=causal, enable_gqa=(self.n_rep > 1))
        out = out.reshape(B, T, -1)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.ffn_hidden_dim)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, cos, sin, mask=None, start_pos=None, use_cache=False):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask, start_pos, use_cache)
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
        
        self.norm = RMSNorm(config.d_model)
        # Weight tying
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        
        cos, sin = precompute_rope_frequencies(config.d_model // config.n_heads, config.ctx_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        
        # Init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        for layer in self.layers:
            layer.attention.reset_cache()

    def forward(self, tokens, start_pos=None, use_cache=False, use_checkpointing=False):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        
        if T > self.config.ctx_len:
            raise ValueError(f"Input sequence length {T} exceeds ctx_len={self.config.ctx_len}")
        if start_pos is None:
            start_pos = 0
        if start_pos + T > self.config.ctx_len:
            raise ValueError(
                f"Requested positions [{start_pos}, {start_pos + T}) exceed ctx_len={self.config.ctx_len}"
            )

        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        
        for layer in self.layers:
            if use_checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, cos, sin, None, start_pos, use_cache, use_reentrant=False
                )
            else:
                h = layer(h, cos, sin, start_pos=start_pos, use_cache=use_cache)
            
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
