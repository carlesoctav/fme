import equinox as eqx
import jax
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert.modeling_bert import BertModel, BertModelForMaskedLM
from src.losses import softmax_cross_entropy_with_integer_labels
import jax.numpy as jnp

def masked_lm_loss_function(model: BertModelForMaskedLM, *args, ignore_index: int = -100):
    """Compute masked LM loss, ignoring positions with label == ignore_index.

    - Vectorizes the model over batch.
    - Uses safe labels for ignored positions to avoid invalid indexing.
    - Normalizes by the number of unmasked tokens.
    """
    x, y = args  
    c, mask = y
    logits = jax.vmap(model)(**x)

    per_token_loss = softmax_cross_entropy_with_integer_labels(logits, c, where= ~mask, reduction = "mean")
    print(f"DEBUGPRINT[187]: train.py:12: per_token_loss={per_token_loss}")

    return per_token_loss



def main():
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
    model = BertModelForMaskedLM(config, key=key, store_config = True)
    model = eqx.nn.inference_mode(model)
    batch_size = 4
    seq_len = 10
    input_ids = jax.random.randint(key, (batch_size, seq_len), minval=0, maxval=config.vocab_size)
    print(f"DEBUGPRINT[190]: train.py:38: input_ids={input_ids}")
    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    print(f"DEBUGPRINT[189]: train.py:39: position_ids={position_ids}")
    token_type_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    print(f"DEBUGPRINT[191]: train.py:42: token_type_ids={token_type_ids}")
    batch_input = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
    }
    batch_labels = jax.random.randint(key, (batch_size, seq_len), minval=0, maxval=config.vocab_size)
    mask = jax.random.uniform(key, (batch_size, seq_len)) < 0.8
    batch_labels = jnp.where(mask, -100, batch_labels)
    print(f"DEBUGPRINT[192]: train.py:51: batch_labels={batch_labels}")
    masked_lm_loss_function(model, batch_input,  (batch_labels, mask))


main()
