from dataclasses import dataclass


@dataclass
class ShiemiConfig:
    # Model architecture
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    vocab_size: int = 4000
    max_seq_length: int = 512
    dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Mixed precision training
    use_mixed_precision: bool = True

    # Optimization
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        # Validate configuration
        assert self.d_ff == 4 * self.d_model, "d_ff should be 4x d_model"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
