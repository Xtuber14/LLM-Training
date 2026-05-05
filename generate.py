import torch
import argparse
from config import nano, small, medium, ModelConfig
from model import Transformer
from tokenizer import Tokenizer

torch.serialization.add_safe_globals([ModelConfig])

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=0.8, top_p=0.95, top_k=50):
    device = next(model.parameters()).device
    model.eval()
    
    tokens = tokenizer.encode(prompt, bos=True)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n{prompt}", end="", flush=True)
    
    # Prefill
    logits = model(x, start_pos=0, use_cache=True)
    next_token_logits = logits[0, -1, :] / temperature
    
    # Sampling logic
    def sample_next_token(logits):
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    next_token = sample_next_token(next_token_logits)
    x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
    print(tokenizer.decode([next_token.item()]), end="", flush=True)
    
    curr_pos = tokens.__len__()
    
    for i in range(max_new_tokens - 1):
        # Forward pass only for the NEW token
        logits = model(next_token.unsqueeze(0), start_pos=curr_pos, use_cache=True)
        next_token_logits = logits[0, -1, :] / temperature
        
        next_token = sample_next_token(next_token_logits)
        curr_pos += 1
        
        # Decode token and print
        decoded = tokenizer.decode([next_token.item()])
        print(decoded, end="", flush=True)
        
        if next_token.item() == tokenizer.sp.eos_id():
            break
            
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Tokenizer()
    actual_vocab_size = tokenizer.vocab_size
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config_name = checkpoint.get("config_name", "nano")
    
    if config_name == "nano":
        config = nano()
    elif config_name == "small":
        config = small()
    else:
        config = medium()
        
    # Override with actual vocab size
    config.vocab_size = actual_vocab_size
        
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint} (Step {checkpoint.get('step', 0)})")
    
    generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
