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
from inference_utils import get_inference_dtype, autocast_context, sample_next_token

torch.serialization.add_safe_globals([ModelConfig])


@torch.no_grad()
def chat_generate(model, tokenizer, conversation_ids, max_new_tokens=512,
                  temperature=0.7, top_p=0.9, top_k=40):
    """
    Generate a response given the full conversation token IDs so far.
    Uses KV cache for efficient autoregressive generation.
    """
    device = next(model.parameters()).device
    dtype = get_inference_dtype(device.type)
    model.eval()
    model.reset_cache()
    
    x = torch.tensor(conversation_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with autocast_context(device.type, dtype):
        logits = model(x, start_pos=0, use_cache=True)
    
    generated_tokens = []
    next_token = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
    
    curr_pos = len(conversation_ids)
    
    for _ in range(max_new_tokens):
        token_id = next_token.item()
        if token_id == tokenizer.eos_id:
            break
        generated_tokens.append(token_id)
        with autocast_context(device.type, dtype):
            logits = model(next_token.unsqueeze(0), start_pos=curr_pos, use_cache=True)
        next_token = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        curr_pos += 1
    
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
        response_text = chat_generate(
            model,
            tokenizer,
            conv_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        print(response_text, end="")
        
        print()  # Newline after response
        
        # Add assistant response to history for multi-turn
        history.append({"role": "assistant", "content": response_text.strip()})
        print()


if __name__ == "__main__":
    main()
