import os
import time
import math
import argparse
import torch
from pathlib import Path

from config import get_train_config, nano, small, medium
from model import Transformer
from tokenizer import Tokenizer
from dataset import get_dataloader
from lr_schedule import get_lr
from checkpoint import save_checkpoint, load_latest_checkpoint

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="nano", choices=["nano", "small", "medium", "colab", "colab_max"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--keep_last", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # Hardware setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Check if bf16 is supported
        if torch.cuda.is_bf16_supported():
            print("BF16 is supported on this GPU.")
            dtype = torch.bfloat16
        else:
            print("BF16 is NOT supported on this GPU. Falling back to FP16.")
            dtype = torch.float16
    else:
        dtype = torch.float32

    # Tokenizer setup
    tokenizer = Tokenizer()
    actual_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {actual_vocab_size}")
    
    # Special tokens for table integration
    try:
        start_table_id = tokenizer.sp.piece_to_id("<|start_table|>")
        end_table_id = tokenizer.sp.piece_to_id("<|end_table|>")
    except:
        start_table_id, end_table_id = -1, -1

    # Model Config
    from config import colab, colab_max
    if args.config == "nano":
        model_config = nano()
    elif args.config == "small":
        model_config = small()
    elif args.config == "colab":
        model_config = colab()
    elif args.config == "colab_max":
        model_config = colab_max()
    else:
        model_config = medium()
        
    # Override with actual vocab size
    model_config.vocab_size = actual_vocab_size
    
    train_config = get_train_config(args.batch_size, args.grad_accum, args.max_steps)
    
    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Dataloader setup
    train_bin = os.path.join(args.data_dir, "data_train.bin")
    if not os.path.exists(train_bin):
        train_bin = os.path.join(args.data_dir, "data.bin")
        if not os.path.exists(train_bin):
             # Try root if data_dir is empty or just a placeholder
             train_bin = "data_train.bin" if os.path.exists("data_train.bin") else "data.bin"
             if not os.path.exists(train_bin):
                raise FileNotFoundError(f"Data not found in {args.data_dir} or root. Run dataset.py first.")
    
    print(f"Loading data from: {train_bin}")
    loader = get_dataloader(train_bin, args.batch_size, model_config.ctx_len)
    
    # Model Setup
    model = Transformer(model_config)
    model.to(device)
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
        
    # Optimizer Setup
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': train_config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas, eps=train_config.eps)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    start_step = 0
    if args.resume:
        from checkpoint import load_latest_checkpoint
        state_dict, opt_dict, ckpt_step, _, ckpt_config = load_latest_checkpoint(ckpt_dir)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(opt_dict)
            start_step = ckpt_step
            print(f"Resumed from step {start_step}")
            
    # Training Loop
    model.train()
    step = start_step
    data_iter = iter(loader)
    
    t0 = time.time()
    while step < train_config.max_steps:
        lr = get_lr(step, train_config.max_steps, train_config.lr, train_config.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(train_config.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
                
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type=device, dtype=dtype):
                logits = model(x, use_checkpointing=args.grad_checkpoint)
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                
                loss = torch.nn.functional.cross_entropy(logits_flat, y_flat, ignore_index=-100, reduction='none')
                
                if start_table_id != -1:
                    is_table_start = (y == start_table_id)
                    is_table_end = (y == end_table_id)
                    table_mask = (torch.cumsum(is_table_start.int(), dim=1) > torch.cumsum(is_table_end.int(), dim=1))
                    table_mask = table_mask | is_table_start | is_table_end
                    
                    weights = torch.ones_like(y, dtype=torch.float32)
                    weights[table_mask] = 2.0
                    loss = loss * weights.view(-1)
                
                loss = loss.mean() / train_config.grad_accum
                
            scaler.scale(loss).backward()
            
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
            
        step += 1
        
        if step % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = (args.batch_size * train_config.grad_accum * model_config.ctx_len * args.log_interval) / dt
            print(f"Step {step} | Loss: {loss.item() * train_config.grad_accum:.4f} | LR: {lr:.2e} | Norm: {grad_norm:.2f} | Tokens/sec: {tokens_per_sec:.0f}")
            t0 = time.time()
            
        if step % args.save_interval == 0:
            from checkpoint import save_checkpoint, cleanup_checkpoints
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, args.config, ckpt_path)
            cleanup_checkpoints(ckpt_dir, keep_last=args.keep_last)

if __name__ == "__main__":
    train()
