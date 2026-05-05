"""
Supervised Fine-Tuning (SFT) Dataset

Supports two JSONL formats:

1. Conversations (multi-turn chat):
   {"conversations": [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "What is 2+2?"},
       {"role": "assistant", "content": "4."}
   ]}

2. Alpaca (single-turn instruction):
   {"instruction": "Explain gravity.", "input": "", "output": "Gravity is..."}

Loss is computed ONLY on assistant response tokens. All prompt/system/user
tokens are masked with -100 so the model learns to generate responses,
not to parrot instructions.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizer import Tokenizer

# ── Chat Template ────────────────────────────────────────────────────────────
# Uses text-based role markers that tokenize naturally with any SentencePiece
# model. No need to retrain the tokenizer with special chat tokens.

ROLE_SYSTEM = "### System:\n"
ROLE_USER = "\n### User:\n"
ROLE_ASSISTANT = "\n### Assistant:\n"

def format_conversation(messages, tokenizer):
    """
    Convert a list of {role, content} messages into token IDs and a loss mask.
    
    Returns:
        input_ids: list[int]  — full tokenized conversation
        labels:    list[int]  — same length; -100 for masked (non-trainable) positions
    """
    input_ids = [tokenizer.bos_id]
    labels = [-100]  # BOS is never a training target
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            # Tokenize role marker + content
            role_tokens = tokenizer.encode(ROLE_SYSTEM)
            content_tokens = tokenizer.encode(content)
            
            # System prompt is context — mask it
            input_ids.extend(role_tokens + content_tokens)
            labels.extend([-100] * (len(role_tokens) + len(content_tokens)))
            
        elif role == "user":
            role_tokens = tokenizer.encode(ROLE_USER)
            content_tokens = tokenizer.encode(content)
            
            # User message is context — mask it
            input_ids.extend(role_tokens + content_tokens)
            labels.extend([-100] * (len(role_tokens) + len(content_tokens)))
            
        elif role == "assistant":
            role_tokens = tokenizer.encode(ROLE_ASSISTANT)
            content_tokens = tokenizer.encode(content)
            
            # The role marker is context — mask it
            input_ids.extend(role_tokens)
            labels.extend([-100] * len(role_tokens))
            
            # The assistant's RESPONSE is the training target — include in loss
            input_ids.extend(content_tokens)
            labels.extend(content_tokens)
            
            # EOS after each assistant turn — also a training target
            input_ids.append(tokenizer.eos_id)
            labels.append(tokenizer.eos_id)
    
    return input_ids, labels


def alpaca_to_conversation(sample):
    """Convert Alpaca format to conversations format."""
    messages = []
    
    # Build instruction with optional input
    instruction = sample["instruction"]
    inp = sample.get("input", "")
    if inp:
        user_content = f"{instruction}\n\n{inp}"
    else:
        user_content = instruction
    
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": sample["output"]})
    
    return messages


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    
    Reads JSONL files and tokenizes conversations with proper loss masking.
    Sequences are truncated or padded to ctx_len.
    """
    
    def __init__(self, data_path, tokenizer, ctx_len, max_samples=None):
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.samples = []
        
        data_path = Path(data_path)
        
        # Support single file or directory of JSONL files
        if data_path.is_dir():
            files = sorted(data_path.glob("*.jsonl"))
        else:
            files = [data_path]
        
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line_num, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON at {f}:{line_num}")
                        continue
                    
                    # Detect format
                    if "conversations" in sample:
                        messages = sample["conversations"]
                    elif "messages" in sample:
                        messages = sample["messages"]
                    elif "instruction" in sample:
                        messages = alpaca_to_conversation(sample)
                    else:
                        print(f"Warning: Unknown format at {f}:{line_num}, skipping")
                        continue
                    
                    self.samples.append(messages)
                    
                    if max_samples and len(self.samples) >= max_samples:
                        break
            
            if max_samples and len(self.samples) >= max_samples:
                break
        
        print(f"Loaded {len(self.samples)} SFT samples from {len(files)} file(s)")
        
        # Print a formatted example for verification
        if self.samples:
            self._print_example(0)
    
    def _print_example(self, idx):
        """Print a formatted example showing what the model will see."""
        messages = self.samples[idx]
        input_ids, labels = format_conversation(messages, self.tokenizer)
        
        # Truncate for display
        display_len = min(len(input_ids), 200)
        
        print(f"\n{'='*60}")
        print(f"Example sample (first {display_len} tokens):")
        print(f"  Total tokens: {len(input_ids)}")
        
        trainable = sum(1 for l in labels if l != -100)
        total = len(labels)
        print(f"  Trainable tokens: {trainable}/{total} ({100*trainable/total:.1f}%)")
        
        # Show decoded text with masking visualization
        text = self.tokenizer.decode(input_ids[:display_len])
        print(f"  Text preview: {text[:300]}...")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        messages = self.samples[idx]
        input_ids, labels = format_conversation(messages, self.tokenizer)
        
        # Truncate to ctx_len + 1 (need one extra for the shifted label)
        if len(input_ids) > self.ctx_len:
            input_ids = input_ids[:self.ctx_len]
            labels = labels[:self.ctx_len]
        
        # For causal LM: x = tokens[:-1], y = tokens[1:]
        # But in SFT, labels are already aligned with input_ids.
        # We want: given input_ids[i], predict labels[i] (which is input_ids[i+1] for assistant tokens)
        # 
        # Actually, the standard approach is:
        #   x = input_ids[:-1]
        #   y = labels[1:]    (shifted by 1 — predicting the next token)
        
        x = input_ids[:-1]
        y = labels[1:]
        
        # Pad to ctx_len - 1 (since we dropped one token for the shift)
        pad_len = (self.ctx_len - 1) - len(x)
        if pad_len > 0:
            x = x + [self.tokenizer.pad_id] * pad_len
            y = y + [-100] * pad_len  # Padding is never a training target
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return x, y


def get_sft_dataloader(data_path, tokenizer, ctx_len, batch_size, 
                       max_samples=None, num_workers=2):
    """Create a DataLoader for SFT training."""
    dataset = SFTDataset(data_path, tokenizer, ctx_len, max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


# ── CLI: Preview a dataset ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preview SFT dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL file or directory")
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()
    
    tokenizer = Tokenizer()
    dataset = SFTDataset(args.data, tokenizer, args.ctx_len, args.max_samples)
    
    for i in range(min(3, len(dataset))):
        x, y = dataset[i]
        trainable = (y != -100).sum().item()
        print(f"Sample {i}: x.shape={x.shape}, trainable_tokens={trainable}/{y.shape[0]}")
