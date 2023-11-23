from typing import Optional
import torch
import torch.nn as nn

from config import Config


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(
            config.n_embed, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embed),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class,
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None):
        # estimte cos sin
        # --------------------

        # forward pass
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)

        x = self.transformer.ln_f(x)  # whyy all of a sudden ln_f in the last??
        return self.lm_head(x)  # (b, t, vocab_size)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = config.norm_class(config.n_embed, eps=config.norm_eps)
        self.ln_2 = (
            None
            if config.shared_attention_norm
            else config.norm_class(config.n_embed, eps=config.norm_eps)
        )
        self.mlp = MLP(config)
        self.config=config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        n_1 = self.ln_1(x)
        h = self.attn(x, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x

        x = h + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
    ):
        pass


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(
            config.intermediate_size, config.n_embed, bias=config.bias
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)
