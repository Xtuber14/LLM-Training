import dataclasses

@dataclasses.dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ctx_len: int
    ffn_hidden_dim: int

@dataclasses.dataclass
class TrainConfig:
    lr: float
    batch_size: int
    grad_accum: int
    max_steps: int
    warmup_steps: int
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    grad_clip: float

def get_ffn_dim(d_model: int) -> int:
    hidden = int((8 / 3) * d_model)
    # Round to nearest multiple of 256
    return ((hidden + 128) // 256) * 256

def nano() -> ModelConfig:
    d_model = 512
    return ModelConfig(
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32000,
        ctx_len=1024,
        ffn_hidden_dim=get_ffn_dim(d_model)
    )

def small() -> ModelConfig:
    d_model = 768
    return ModelConfig(
        d_model=d_model,
        n_layers=12,
        n_heads=12,
        n_kv_heads=3,  # 12 / 4 = 3
        vocab_size=32000,
        ctx_len=2048,
        ffn_hidden_dim=get_ffn_dim(d_model)
    )

def medium() -> ModelConfig:
    d_model = 1024
    return ModelConfig(
        d_model=d_model,
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,  # 16 / 4 = 4
        vocab_size=32000,
        ctx_len=2048,
        ffn_hidden_dim=get_ffn_dim(d_model)
    )

def get_train_config(batch_size: int, grad_accum: int, max_steps: int) -> TrainConfig:
    return TrainConfig(
        lr=3e-4,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        warmup_steps=200,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        grad_clip=1.0
    )
