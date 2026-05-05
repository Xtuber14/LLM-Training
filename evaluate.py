import torch
import math
from dataset import MemmapDataset
from torch.utils.data import DataLoader
from model import Transformer
from config import nano, small, medium, ModelConfig

torch.serialization.add_safe_globals([ModelConfig])
import argparse

@torch.no_grad()
def evaluate_perplexity(model, data_file, batch_size=4):
    device = next(model.parameters()).device
    model.eval()
    
    # Use context len from config
    ctx_len = model.config.ctx_len
    
    dataset = MemmapDataset(data_file, ctx_len=ctx_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        
        # Reshape for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        
        # Calculate loss (sum reduction to count tokens properly)
        loss = torch.nn.functional.cross_entropy(logits_flat, y_flat, ignore_index=-100, reduction='sum')
        
        valid_tokens = (y_flat != -100).sum().item()
        
        total_loss += loss.item()
        total_tokens += valid_tokens
        
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    from tokenizer import Tokenizer
    tokenizer = Tokenizer()
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint.get('config')
    if config is None:
        # Fallback to config_name
        config_name = checkpoint.get("config_name", "nano")
        if config_name == "nano":
            config = nano()
        elif config_name == "small":
            config = small()
        else:
            config = medium()
    
    # Ensure vocab size matches tokenizer
    config.vocab_size = tokenizer.vocab_size
        
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Evaluating perplexity on {args.data_file}...")
    with torch.autocast(device_type=device, dtype=dtype):
        ppl, loss = evaluate_perplexity(model, args.data_file)
    print(f"Cross Entropy Loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")
