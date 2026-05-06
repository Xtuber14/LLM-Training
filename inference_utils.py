import contextlib
import torch


def get_inference_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_inference_dtype(device):
    if device != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device, dtype):
    if device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    filtered = logits.clone()

    if top_k > 0:
        values, _ = torch.topk(filtered, min(top_k, filtered.size(-1)))
        threshold = values[..., -1, None]
        filtered[filtered < threshold] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered[indices_to_remove] = float("-inf")

    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
