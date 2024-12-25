import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

from .attention import CausalSelfAttention


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (var + self.eps).sqrt() + self.bias


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Pre-norm architecture
        h = self.ln1(x)
        h = self.attn(h, attention_mask)
        x = x + h

        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h

        return x


class ShiemiTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_seq_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([DecoderBlock(config)
                                    for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config.d_model)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        b, t = input_ids.size()
        assert t <= self.config.max_seq_length, f"Input sequence length ({
            t}) exceeds model's maximum ({self.config.max_seq_length})"

        # Get positions for embedding
        pos = torch.arange(0, t, dtype=torch.long,
                           device=input_ids.device).unsqueeze(0)

        # Token + Position embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(pos)
        x = self.dropout(token_embeddings + position_embeddings)

        # Transform through decoder blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_final(x)

        # Use the token embedding weights for the output projection (weight tying)
        logits = F.linear(x, self.token_embedding.weight)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate):
        # Separate weight decay from bias and layer norm terms
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, nn.Embedding)
        blacklist_weight_modules = (LayerNorm,)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate that we've got all parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {
            inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {
            param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(optim_groups, lr=learning_rate)
