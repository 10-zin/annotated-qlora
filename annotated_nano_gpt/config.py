from dataclasses import dataclass
import torch
from torch.nn.modules.normalization import LayerNorm


@dataclass
class Config:
    name: str = ""
    block_size: int = 2048
    n_head: int = 8
    n_query_groups: int = n_head
    n_layer: int = 6
    vocab_size: int = 50304
    n_embed: int = 512
    head_size: int = n_embed // n_head
    padded_vocab_size: int = 50304
    norm_eps: float = 1e-5
    lm_head_bias: bool = False  # why no bias ?? not sure
    shared_attention_norm: bool = False
    intermediate_size: int = 4 * n_embed
    bias: bool = True
    gelu_approximate: str = "none"
    parallel_residual: bool = True
    rotary_percentage: float=0.25
    rope_n_elem = int(rotary_percentage * head_size)


    norm_class: LayerNorm = LayerNorm
