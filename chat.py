"""
Interactive Chat with an SFT-trained model.

Uses the same chat template as SFT training so the model sees
the exact format it was fine-tuned on.

Usage:
    python chat.py --checkpoint checkpoints_sft/sft_final.pt
    python chat.py --checkpoint checkpoints_sft/sft_final.pt --system "You are a pirate."
"""

import torch
import argparse
from config import nano, small, medium, ModelConfig
from model import Transformer
from tokenizer import Tokenizer
from sft_dataset import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT

torch.serialization.add_safe_globals([ModelConfig])


@torch.no_grad()
def chat_generate(model, tokenizer, conversation_ids, max_new_tokens=512,
                  temperature=0.7, top_p=0.9, top_k=40):
    """
    Generate a response given the full conversation token IDs so far.
    Uses KV cache for efficient autoregressive generation.
    """
    device = next(model.parameters()).device
    model.eval()
    
    x = torch.tensor(conversation_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Prefill the full conversation context
    logits = model(x, start_pos=0, use_cache=True)
    next_token_logits = logits[0, -1, :] / temperature
    
    def sample(logits):
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    generated_tokens = []
    next_token = sample(next_token_logits)
    generated_tokens.append(next_token.item())
    
    curr_pos = len(conversation_ids)
    
    for _ in range(max_new_tokens - 1):
        logits = model(next_token.unsqueeze(0), start_pos=curr_pos, use_cache=True)
        next_token_logits = logits[0, -1, :] / temperature
        next_token = sample(next_token_logits)
        
        token_id = next_token.item()
        generated_tokens.append(token_id)
        curr_pos += 1
        
        # Stop on EOS
        if token_id == tokenizer.eos_id:
            break
    
    # Decode, stripping the EOS token from output
    if generated_tokens and generated_tokens[-1] == tokenizer.eos_id:
        generated_tokens = generated_tokens[:-1]
    
    return tokenizer.decode(generated_tokens)


def build_conversation_ids(tokenizer, messages):
    """
    Tokenize a full conversation history into IDs using the chat template.
    Ends with the assistant role marker so the model generates the response.
    """
    ids = [tokenizer.bos_id]
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            ids.extend(tokenizer.encode(ROLE_SYSTEM + content))
        elif role == "user":
            ids.extend(tokenizer.encode(ROLE_USER + content))
        elif role == "assistant":
            ids.extend(tokenizer.encode(ROLE_ASSISTANT + content))
            ids.append(tokenizer.eos_id)
    
    # Add the assistant marker to prompt generation
    ids.extend(tokenizer.encode(ROLE_ASSISTANT))
    
    return ids


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system", type=str, default="You are a helpful assistant.",
                        help="System prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    config = checkpoint.get('config')
    if config is None:
        config_name = checkpoint.get('config_name', 'nano')
        if config_name == 'nano':
            config = nano()
        elif config_name == 'small':
            config = small()
        else:
            config = medium()
    
    config.vocab_size = tokenizer.vocab_size
    
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    step = checkpoint.get('step', 0)
    print(f"Loaded model (step {step})")
    print(f"System: {args.system}")
    print(f"Type 'quit' or 'exit' to end. Type 'clear' to reset conversation.\n")

    # ── Chat Loop ────────────────────────────────────────────────────────
    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history = []
            if args.system:
                history.append({"role": "system", "content": args.system})
            print("-- Conversation cleared --\n")
            continue
        
        # Add user message
        history.append({"role": "user", "content": user_input})
        
        # Build token IDs for the full conversation
        conv_ids = build_conversation_ids(tokenizer, history)
        
        # Check context length
        if len(conv_ids) > config.ctx_len - 100:
            print(f"[Warning: conversation is {len(conv_ids)} tokens, "
                  f"approaching ctx_len={config.ctx_len}. Consider clearing.]")
        
        # Generate
        print("Assistant: ", end="", flush=True)
        
        # For streaming, we generate token by token
        device = next(model.parameters()).device
        model.eval()
        
        x = torch.tensor(conv_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x, start_pos=0, use_cache=True)
        next_token_logits = logits[0, -1, :] / args.temperature
        
        def sample_token(lgt):
            if args.top_k > 0:
                v, _ = torch.topk(lgt, min(args.top_k, lgt.size(-1)))
                lgt[lgt < v[-1]] = -float('Inf')
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(lgt, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                lgt[indices_to_remove] = -float('Inf')
            probs = torch.softmax(lgt, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        
        generated_tokens = []
        next_token = sample_token(next_token_logits)
        curr_pos = len(conv_ids)
        
        response_text = ""
        for _ in range(args.max_new_tokens):
            token_id = next_token.item()
            
            if token_id == tokenizer.eos_id:
                break
            
            generated_tokens.append(token_id)
            decoded = tokenizer.decode([token_id])
            print(decoded, end="", flush=True)
            response_text += decoded
            
            curr_pos += 1
            logits = model(next_token.unsqueeze(0), start_pos=curr_pos - 1, use_cache=True)
            next_token_logits = logits[0, -1, :] / args.temperature
            next_token = sample_token(next_token_logits)
        
        print()  # Newline after response
        
        # Add assistant response to history for multi-turn
        history.append({"role": "assistant", "content": response_text.strip()})
        print()


if __name__ == "__main__":
    main()
