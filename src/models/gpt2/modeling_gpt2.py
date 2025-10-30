import equinox as eqx
import jax
import jax.numpy as jnp
from transformers import GPT2Config

from src import nn


class GPT2MLP(eqx.Module):
    c_fc: nn.Linear
    c_proj: nn.Linear
    act_fn: eqx.field(static = True)


    def __init__(
        self,
        config: GPT2Config,
        *,
        rngs
    ):
        self.c_fc = nn.Linear(config.n_embd, config.n_inner & config.n_embd, rngs = rngs)
        self.c_proj = nn.Linear(config.n_inner * config.n_embd, config.n_embd, rngs = rngs)
        self.act_fn = jax.nn.gelu 


    def __call__(self, hidden_states, *, rngs):
        x = self.c_fc(hidden_states, rngs = rngs)
        x = self.act_fn(x)
        x = self.c_proj(x, rngs = rngs)
        return x


class GPT2Attention(eqx.Module):
    c_attn: nn.Linear
    c_proj: nn.Linear
    attn_module: eqx.Module
    embed_dim: int = eqx.field(static = True)
    num_heads: int = eqx.field(static = True)
    head_size: int = eqx.field(static = True)


    def __init__(self, config, *, rngs):
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_size = self.embed_dim // self.num_heads

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, rngs = rngs)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, rngs = rngs)
        self.attn_module = nn.make_attention_module(config)


    def __call__(self, hidden_states, attention_mask, *, rngs):
        B, T, H = hidden_states.shape
        qkv_out = self.c_attn(hidden_states, rngs = rngs).reshape(B, T, 3, self.num_heads, self.head_size)
        q, k, v = jnp.split(qkv_out, 3, 2)
        attn_output = self.attn_module(
            q,
            k,
            v,
            mask = attention_mask,
            rngs = rngs
        )

        attn_output = self.c_proj(attn_output)
        return attn_output


class Gpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: GPT2Attention 
    ln_2: nn.LayerNorm
    mlp: GPT2MLP
    resid_dropout: nn.Dropout



    def __init__(self, config, *, rngs):
        self.ln_1 = nn.LayerNorm(config.n_embd, rngs = rngs)
        self.ln_2 = nn.LayerNorm(config.n_embd, rngs = rngs)
        self.mlp = GPT2MLP(config, rngs = rngs) 
        self.attn = GPT2Attention(config, rngs = rngs)
        self.resid_dropout = nn.Dropout(config.resid_pdrop, rngs = rngs)


    def __call__(self, hidden_states, attention_mask, *, rngs):
        attn_output = self.attn(self.ln_1(hidden_states), attention_mask, rngs = rngs)
        attn_output = self.resid_dropout(attn_output, rngs = rngs)
        hidden_states = hidden_states + attn_output

        ff_output = self.mlp(self.ln_2(hidden_states), rngs = rngs)
        ff_output = self.resid_dropout(ff_output, rngs = rngs) 
        hidden_states = hidden_states + ff_output
        return hidden_states


