"""
Supervised Fine-Tuning (SFT) Training Script

Loads a pretrained checkpoint and fine-tunes it on instruction/conversation data.
Uses lower learning rates, proper loss masking, and SFT-specific hyperparameters.

Usage:
    python sft_train.py \
        --checkpoint checkpoints/step_5000.pt \
        --data sft_data/train.jsonl \
        --batch_size 2 \
        --grad_accum 8 \
        --max_steps 2000 \
        --lr 2e-5
"""

import os
import time
import math
import argparse
import torch
from pathlib import Path
import inspect

from config import get_train_config, nano, small, medium, ModelConfig
from model import Transformer
from tokenizer import Tokenizer
from sft_dataset import get_sft_dataloader, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT
from lr_schedule import get_lr
from checkpoint import save_checkpoint, load_latest_checkpoint
from inference_utils import autocast_context

torch.serialization.add_safe_globals([ModelConfig])

def build_optimizer(optim_groups, lr, betas, eps):
    kwargs = {"lr": lr, "betas": betas, "eps": eps}
    if "fused" in inspect.signature(torch.optim.AdamW).parameters and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(optim_groups, **kwargs)


def configure_training_backends(device):
    if device != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def sft_train():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to JSONL file or directory of JSONL files")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Optional validation JSONL file")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Peak learning rate (SFT uses much lower LR than pretraining)")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of training samples (for testing)")
    parser.add_argument("--output_dir", type=str, default="checkpoints_sft",
                        help="Directory to save SFT checkpoints")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    # ── Hardware ─────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    configure_training_backends(device)
    
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.2f} GB")
        
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using BF16 mixed precision")
        else:
            dtype = torch.float16
            print("Using FP16 mixed precision")
    else:
        dtype = torch.float32

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ── Load Pretrained Checkpoint ───────────────────────────────────────
    print(f"\nLoading pretrained checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    # Get config from checkpoint
    config = checkpoint.get('config')
    if config is None:
        config_name = checkpoint.get('config_name', 'nano')
        if config_name == 'nano':
            config = nano()
        elif config_name == 'small':
            config = small()
        else:
            config = medium()
    
    # Override vocab size to match tokenizer
    config.vocab_size = tokenizer.vocab_size
    
    pretrain_step = checkpoint.get('step', 0)
    print(f"  Pretrained at step: {pretrain_step}")
    print(f"  Model config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, ctx_len={config.ctx_len}")

    # ── Model ────────────────────────────────────────────────────────────
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── SFT Dataset ──────────────────────────────────────────────────────
    print(f"\nLoading SFT data from: {args.data}")
    train_loader = get_sft_dataloader(
        args.data, tokenizer, config.ctx_len, args.batch_size, 
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    
    val_loader = None
    if args.val_data:
        print(f"Loading validation data from: {args.val_data}")
        val_loader = get_sft_dataloader(
            args.val_data, tokenizer, config.ctx_len, 
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    # ── Optimizer (SFT-specific hyperparameters) ─────────────────────────
    # Key differences from pretraining:
    #   - Much lower learning rate (2e-5 vs 3e-4)
    #   - Lower weight decay (0.01 vs 0.1) to preserve pretrained features
    #   - Same Adam betas work well
    
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = build_optimizer(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    
    # Mixed precision scaler (only for FP16, not BF16)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    # ── Checkpoint directory ─────────────────────────────────────────────
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ── Evaluation function ──────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(loader, max_batches=50):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        
        for x, y in loader:
            if batches >= max_batches:
                break
            x = x.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            
            with autocast_context(device, dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100
                )
            
            # Count only trainable tokens for accurate loss
            n_tokens = (y != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            batches += 1
        
        model.train()
        if total_tokens == 0:
            return float('inf')
        return total_loss / total_tokens

    # ── Training Loop ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting SFT Training")
    print(f"  Steps: {args.max_steps}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Gradient checkpointing: {args.grad_checkpoint}")
    print(f"{'='*60}\n")

    model.train()
    step = 0
    best_val_loss = float('inf')
    data_iter = iter(train_loader)
    
    t0 = time.time()
    while step < args.max_steps:
        # Cosine decay with warmup — using lower SFT learning rate
        lr = get_lr(step, args.max_steps, args.lr, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation
        accum_loss = 0.0
        accum_tokens = 0
        
        for micro_step in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            
            x = x.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            
            with autocast_context(device, dtype):
                logits = model(x, use_checkpointing=args.grad_checkpoint)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100
                )
                scaled_loss = loss / args.grad_accum
            
            scaler.scale(scaled_loss).backward()
            
            # Track actual loss on trainable tokens
            n_tokens = (y != -100).sum().item()
            accum_loss += loss.item() * n_tokens
            accum_tokens += n_tokens
        
        # Optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if step % args.log_interval == 0:
                print(f"Step {step} | Skipping update due to non-finite grad norm: {grad_norm}")
            continue
        scaler.step(optimizer)
        scaler.update()
        
        step += 1
        
        # ── Logging ──────────────────────────────────────────────────
        if step % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            
            avg_loss = accum_loss / max(accum_tokens, 1)
            tokens_per_sec = (accum_tokens * args.log_interval) / dt
            
            print(f"Step {step:5d}/{args.max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Norm: {grad_norm:.2f} | "
                  f"Tok/s: {tokens_per_sec:.0f}")
            t0 = time.time()
        
        # ── Validation ───────────────────────────────────────────────
        if val_loader and step % args.eval_interval == 0:
            val_loss = evaluate(val_loader)
            print(f"  ── Val Loss: {val_loss:.4f} {'(best!)' if val_loss < best_val_loss else ''}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = ckpt_dir / "best_sft.pt"
                save_checkpoint(model, optimizer, step, "sft", best_path)
        
        # ── Save checkpoint ──────────────────────────────────────────
        if step % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"sft_step_{step}.pt"
            save_checkpoint(model, optimizer, step, "sft", ckpt_path)
    
    # Save final checkpoint
    final_path = ckpt_dir / "sft_final.pt"
    save_checkpoint(model, optimizer, step, "sft", final_path)
    print(f"\nSFT training complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    sft_train()
