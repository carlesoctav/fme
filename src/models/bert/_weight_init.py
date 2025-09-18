from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal, zeros as zeros_init, ones as ones_init

from ... import nn
from ..._darray import DArray


class BertModelWeightPlanMixin:
    """
    Provides `init_weights_plan(self, module, key)` following HF BERT conventions:
    - Linear: weight ~ N(0, initializer_range), bias zeros
    - Embedding: weight ~ N(0, initializer_range); if pad_token_id is set, zero that row
    - LayerNorm: weight ones, bias zeros
    """

    def init_weights_plan(self, module, key):  # noqa: D401
        cfg = getattr(self, "config", None)
        std = getattr(cfg, "initializer_range", 0.02)
        pad_idx = getattr(cfg, "pad_token_id", None)

        # Linear
        if isinstance(module, nn.Linear):
            wkey, bkey = jax.random.split(key, 2)
            w_shape = (module.out_features, module.in_features)
            w_dtype = module.params_dtype
            new_w = normal(std)(wkey, w_shape, dtype=w_dtype)
            new_bias = None
            if module.use_bias and module.bias is not None:
                b_shape = (module.out_features,)
                new_bias = zeros_init(bkey, b_shape, dtype=w_dtype)
            new_mod = module
            new_mod = eqx.tree_at(lambda m: m.weight, new_mod, DArray(value=new_w, pspec=module.weight.pspec))
            if module.use_bias and module.bias is not None:
                new_mod = eqx.tree_at(lambda m: m.bias, new_mod, DArray(value=new_bias, pspec=module.bias.pspec))
            return new_mod

        # Embedding
        if isinstance(module, nn.Embedding):
            w_shape = (module.num_embeddings, module.embedding_dim)
            w_dtype = module.params_dtype
            new_w = normal(std)(key, w_shape, dtype=w_dtype)
            if pad_idx is not None and 0 <= int(pad_idx) < module.num_embeddings:
                new_w = new_w.at[int(pad_idx)].set(jnp.zeros((module.embedding_dim,), dtype=w_dtype))
            return eqx.tree_at(lambda m: m.weight, module, DArray(value=new_w, pspec=module.weight.pspec))

        # LayerNorm
        if isinstance(module, nn.LayerNorm):
            w_shape = module.normalized_shape
            w_dtype = module.params_dtype
            new_w = ones_init(jax.random.PRNGKey(0), w_shape, dtype=w_dtype)
            new_mod = eqx.tree_at(lambda m: m.weight, module, DArray(value=new_w, pspec=module.weight.pspec if module.weight is not None else None))
            if module.bias is not None:
                new_b = zeros_init(jax.random.PRNGKey(0), w_shape, dtype=w_dtype)
                new_mod = eqx.tree_at(lambda m: m.bias, new_mod, DArray(value=new_b, pspec=module.bias.pspec))
            return new_mod

        # Default: leave unchanged
        return module

