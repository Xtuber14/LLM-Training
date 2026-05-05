import torch
import numpy as np
import pytest
from tokenizer import Tokenizer, train_tokenizer
from model import precompute_freqs_cis, apply_rotary_emb, Transformer, RMSNorm, SwiGLU
from config import nano
from attention import flash_attention

def test_rope_frequencies():
    dim = 64
    end = 128
    theta = 10000.0
    
    freqs_cis = precompute_freqs_cis(dim, end, theta)
    
    assert freqs_cis.shape == (end, dim // 2)
    
    # Check frequency values
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    expected_freqs = torch.outer(t, freqs)
    expected_cis = torch.polar(torch.ones_like(expected_freqs), expected_freqs)
    
    assert torch.allclose(freqs_cis, expected_cis, atol=1e-5)

def test_rope_application():
    batch, seq, heads, dim = 2, 10, 4, 32
    xq = torch.randn(batch, seq, heads, dim)
    xk = torch.randn(batch, seq, heads, dim)
    
    freqs_cis = precompute_freqs_cis(dim, seq)
    
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    
    assert xq_rot.shape == xq.shape
    assert xk_rot.shape == xk.shape

def test_rmsnorm():
    dim = 128
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    
    assert out.shape == x.shape
    # Check if RMS is close to 1
    rms = torch.sqrt(torch.mean(out**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

def test_swiglu():
    d_model = 128
    hidden = 256
    ffn = SwiGLU(d_model, hidden)
    x = torch.randn(2, 10, d_model)
    out = ffn(x)
    assert out.shape == x.shape

def test_attention_shape():
    B, T, H, D = 2, 64, 4, 32
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    
    out = flash_attention(q, k, v, causal=True)
    assert out.shape == (B, T, H, D)

def test_swiglu_logic():
    d_model = 128
    hidden = 256
    ffn = SwiGLU(d_model, hidden)
    x = torch.randn(2, 10, d_model)
    
    # Manually compute expected output with correct order
    # return self.w3(F.silu(self.w1(x)) * self.w2(x))
    w1_out = ffn.w1(x)
    w2_out = ffn.w2(x)
    expected = ffn.w3(torch.nn.functional.silu(w1_out) * w2_out)
    
    out = ffn(x)
    assert torch.allclose(out, expected)

def test_kv_cache_correctness():
    config = nano()
    config.n_layers = 2
    config.d_model = 64
    config.n_heads = 4
    config.vocab_size = 100
    
    model = Transformer(config)
    model.eval()
    
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # 1. Full sequence forward
    with torch.no_grad():
        full_logits = model(prompt_ids)
        full_last_logit = full_logits[:, -1, :]
        
    # 2. KV cache forward
    with torch.no_grad():
        # Prefill
        model(prompt_ids[:, :-1], start_pos=0, use_cache=True)
        # Single step
        cache_logits = model(prompt_ids[:, -1:], start_pos=9, use_cache=True)
        cache_last_logit = cache_logits[:, -1, :]
        
    assert torch.allclose(full_last_logit, cache_last_logit, atol=1e-5)

def test_transformer_forward():
    config = nano()
    # reduce sizes for fast test
    config.n_layers = 2
    config.d_model = 128
    config.n_heads = 4
    config.n_kv_heads = 2
    config.ffn_hidden_dim = 256
    config.vocab_size = 1000
    config.ctx_len = 64
    
    model = Transformer(config)
    x = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(x)
    
    assert logits.shape == (2, 32, config.vocab_size)

def test_loss_computation():
    config = nano()
    config.n_layers = 1
    config.vocab_size = 100
    model = Transformer(config)
    
    x = torch.randint(0, config.vocab_size, (2, 10))
    y = torch.randint(0, config.vocab_size, (2, 10))
    y[0, 5:] = -100 # padding
    
    logits = model(x)
    logits_flat = logits.view(-1, logits.size(-1))
    y_flat = y.view(-1)
    
    loss = torch.nn.functional.cross_entropy(logits_flat, y_flat, ignore_index=-100)
    assert loss.item() > 0
    assert not torch.isnan(loss)
