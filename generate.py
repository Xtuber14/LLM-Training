import torch
import argparse
from config import nano, small, medium, colab, colab_max, ModelConfig
from model import Transformer
from tokenizer import Tokenizer
from inference_utils import get_inference_dtype, autocast_context, sample_next_token
torch.serialization.add_safe_globals([ModelConfig])

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=0.8, top_p=0.95, top_k=50):
    device = next(model.parameters()).device
    dtype = get_inference_dtype(device.type)
    model.eval()
    model.reset_cache()

    tokens = tokenizer.encode(prompt, bos=True)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n{prompt}", end="", flush=True)

    with autocast_context(device.type, dtype):
        logits = model(x, start_pos=0, use_cache=True)

    next_token = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
    curr_pos = len(tokens)

    for _ in range(max_new_tokens):
        token_id = next_token.item()
        if token_id == tokenizer.eos_id:
            break

        print(tokenizer.decode([token_id]), end="", flush=True)

        with autocast_context(device.type, dtype):
            logits = model(next_token.unsqueeze(0), start_pos=curr_pos, use_cache=True)
        next_token = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        curr_pos += 1

    print("\n")


def auto_detect_config(state_dict):
    """Detect model config from checkpoint weights."""
    dim = state_dict['tok_embeddings.weight'].shape[1]
    n_layers = max(int(k.split('.')[1]) for k in state_dict if k.startswith('layers.')) + 1
    return dim, n_layers

CONFIG_MAP = {
    "nano": nano,
    "small": small,
    "medium": medium,
    "colab": colab,
    "colab_max": colab_max,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cuda', 'cpu'. Auto-detects if not set.")
    parser.add_argument("--half", action="store_true",
                        help="Load model in float16 to reduce VRAM usage.")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = Tokenizer()
    actual_vocab_size = tokenizer.vocab_size

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Strip _orig_mod. prefix if model was saved with torch.compile()
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Detected torch.compile() checkpoint, stripping '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Resolve config: use saved name, or auto-detect from weights
    config_name = checkpoint.get("config_name", "")
    if config_name in CONFIG_MAP:
        config = CONFIG_MAP[config_name]()
        print(f"Using saved config: '{config_name}'")
    else:
        dim, n_layers = auto_detect_config(state_dict)
        print(f"Config name '{config_name}' not recognized — auto-detected: dim={dim}, n_layers={n_layers}")
        detected = None
        for name, fn in CONFIG_MAP.items():
            c = fn()
            if c.d_model == dim and c.n_layers == n_layers:
                detected = name
                config = c
                break
        if detected:
            print(f"Matched to config: '{detected}'")
        else:
            raise ValueError(
                f"No matching config found for dim={dim}, n_layers={n_layers}. "
                f"Please add a matching config to config.py."
            )

    # Override with actual vocab size from tokenizer
    config.vocab_size = actual_vocab_size

    model = Transformer(config)
    model.load_state_dict(state_dict)

    # Use float16 to reduce VRAM (auto-enabled on cuda unless --device cpu)
    if args.half or (device == "cuda" and args.device != "cpu"):
        print("Converting model to float16 to save VRAM...")
        model = model.half()

    model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint} (Step {checkpoint.get('step', 0)})")

    generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
