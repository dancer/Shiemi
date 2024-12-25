import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        # Single matrix for Q, K, V projections to save memory
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Causal mask to ensure we only attend to previous tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_length, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d',
                            h=self.n_heads, qkv=3).unbind(0)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal masking
        causal_mask = self.mask[:, :, :seq_length, :seq_length]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Optional padding mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Combine heads
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.proj(out)
