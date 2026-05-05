import math

def get_lr(step: int, max_steps: int, lr: float, warmup_steps: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    """
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return lr * 0.1
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr_min = lr * 0.1
    return lr_min + coeff * (lr - lr_min)
