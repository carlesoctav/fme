import equinox as eqx
import jax
import jax.numpy as jnp
from transformers.models.bert.configuration_bert import BertConfig

from src.models.bert.modeling_bert import BertForMaskedLM


def main():
    # Tiny config to keep things fast and simple
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    )

    key = jax.random.PRNGKey(0)
    model = BertForMaskedLM(config, key=key)

    seq_len = 10
    input_ids = jax.random.randint(key, (seq_len,), minval=0, maxval=config.vocab_size)
    position_ids = jnp.arange(seq_len)
    token_type_ids = jnp.zeros((seq_len,), dtype=jnp.int32)

    # Run a simple forward pass in inference mode (no dropout randomness)
    model = eqx.nn.inference_mode(model)
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        key=None,
    )

    print("Logits shape:", logits.shape)
    assert logits.shape == (seq_len, config.vocab_size)


if __name__ == "__main__":
    main()

