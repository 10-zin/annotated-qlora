import math
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
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.attn = nn.Linear(config.n_embed, shape, bias=config.bias)
        self.kv_cache: Optional[KVCache] = None
        self.config=config

    def get_q_k_v_less_complicated(self, qkv, B, T, input_pos):
        # qkv : B, T, Shape

        # nh_qkv = n_h+2*n_q_grps
        # qkv : B, T, nh_qkv, head_size
        qkv = qkv.view(B, T, self.config.n_head + 2 * self.config.n_query_groups, self.config.head_size)

        # qkv: B, nh_qkv, T, head_size
        qkv = qkv.permute(0,2,1,3) 

        # q: B, n_head, T, head_size
        # k: B, n_q_grps, T, head_size 
        # v: B, n_q_grps, T, head_size
        q, k, v = qkv.split((self.config.n_head, self.config.n_query_groups, self.config.n_query_groups), dim=1)

        # R-TODO: find a way to shorten this too.. expand.. what does it do? and when is this being used.
        q_per_group = self.config.n_head//self.config.n_query_groups
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_group, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_group, T, self.config.head_size)

        return q, k, v

    def get_q_k_v(self, qkv, B, T, input_pos):
        q_per_group = self.config.n_head//self.config.n_query_groups
        qkv_per_group = q_per_group + 2 # +2 -> 1 key and 1 value per group
        qkv = qkv.view(B, T, self.config.n_query_groups, qkv_per_group, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)

        q, k, v = qkv.split((q_per_group, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_group, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_group, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
        
        return q, k, v
    
    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # R-TODO: Deep Dive
        # 1. intuition behind scale
        # 2. Soft attention = system 1 -> whats a system 2 way -> for llm to do to do more planning based understanding.
            # 2a. can you give the architecture tools to plan.
            # 2b. 
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def forward(
        self, x, cos, sin, mask, input_pos
    ):
        B, T, C = x.size()
        qkv = self.attn(x)

        q, k, v = self.get_q_k_v(qkv, B, T, input_pos)
        # q, k, v = self.get_q_k_v_less_complicated(qkv, B, T, input_pos)


        # R-TODO: Investigate rope
        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)
    
    def build_kv_cache(self, batch_size, max_seq_length, rope_cache_length, device, dtype) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.coos-size(-1)` value")
            k_shape = v_shape
        else:
            # R-TODO: Understand this-> rope_cache_length+self.config.head_size-self.config.rope_n_elem
            k_shape = (batch_size, heads, max_seq_length, rope_cache_length+self.config.head_size-self.config.rope_n_elem)
        
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)



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


class KVCache(nn.Module):
    def __init__(self, k_shape, v_shape, device, dtype):
        super().__init__()

        # R-TODO: Why registering only k,v why not q. Understand kv cache better. Limitations of KV cache -> Improvements.
        self.register_buffer('k', torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer('v', torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k, v):
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)

        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)



