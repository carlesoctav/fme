"""
Benchmark jitted initialization of BERT 2.5B model (Equinox with fold_in).

This measures compilation time by wrapping init in @jax.jit.
"""

import time
import jax
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel


def create_bert_large_config():
    return BertConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=10240,
        max_position_embeddings=512,
        vocab_size=30522,
        type_vocab_size=2,
        _attn_implementation="sdpa",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def main():
    config = create_bert_large_config()
    key = jax.random.key(10)

    @jax.jit
    def init_model():
        return BertModel(config, key=key)

    print("Running jitted init (measures compilation + init)...")
    start = time.monotonic()
    module = init_model()
    jax.block_until_ready(jax.tree.leaves(module))
    diff = time.monotonic() - start
    print(f"Equinox jitted init time: {diff:.2f}s")


if __name__ == "__main__":
    main()
