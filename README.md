# LLM From Scratch (ROCm & PyTorch 2.x)

This is a complete, production-quality implementation of a decoder-only large language model from absolute zero, built natively for AMD Radeon RX 6650 XT (8 GB VRAM, RDNA2).

## Features
- **Flash Attention 2** via xformers (or chunked attention fallback)
- **Rotary Position Embeddings (RoPE)**
- **SwiGLU** Feed-Forward Networks
- **RMSNorm** (Pre-Norm)
- **Grouped-Query Attention (GQA)** (4:1 ratio)
- Custom BPE Tokenizer training (SentencePiece)
- Mixed Precision (BF16 native on RDNA2)
- Gradient Checkpointing & Accumulation
- Clean, hand-written training loop with AdamW and Cosine Decay

## Requirements
- Python 3.11+ (Conda environment recommended)
- AMD ROCm 6.2+ drivers installed
- Packages: see `requirements.txt`

## VRAM Budget Analysis (AdamW + BF16)
- **Nano** (d=512, L=8, ctx=1024, bs=4): ~920 MB -> fits easily in 8GB
- **Small** (d=768, L=12, ctx=2048, bs=4): ~3.0 GB -> fits comfortably in 8GB
- **Medium** (d=1024, L=16, ctx=2048, bs=4): ~6.3 GB -> tight, uses gradient checkpointing

## Setup Instructions for ROCm

Create a Conda environment and install PyTorch with ROCm support:
```bash
conda create -n llm python=3.11 -y
conda activate llm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install sentencepiece numpy tqdm xformers pytest
```

## Usage

### 1. Train Tokenizer & Prepare Data
Place your markdown files in `training_data/`. We have provided `training_data/sample.md` for testing.
```bash
# Train BPE Tokenizer
python tokenizer.py --data_dir training_data/ --vocab_size 32000

# Chunk text and build memory-mapped dataset
python dataset.py
```

### 2. Training
```bash
python train.py \
  --config nano \
  --data_dir training_data/ \
  --batch_size 4 \
  --grad_accum 4 \
  --max_steps 50000 \
  --grad_checkpoint \
  --compile
```

### 3. Generation (Inference)
```bash
python generate.py \
  --checkpoint checkpoints/step_500.pt \
  --prompt "The quick brown fox" \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.95
```

### 4. Perplexity Evaluation
```bash
python evaluate.py --checkpoint checkpoints/step_500.pt --data_file training_data/data_val.bin
```

### 5. Running Tests
```bash
python -m pytest tests/test_model.py -v
```

## Architectural Decisions

- **RoPE**: Generalizes well to unseen sequence lengths and doesn't require extra learnable parameters. We use the analytical formula.
- **SwiGLU**: LLaMA/PaLM showed SwiGLU improves quality per FLOP over GELU. Gated activation provides more expressive capacity.
- **RMSNorm**: Removes mean-centering to save compute. Empirically matches LayerNorm quality but simpler.
- **GQA**: The 4:1 ratio balances quality (close to Multi-Head Attention) with memory efficiency (4× less KV cache), which is critical for 8GB VRAM cards.
