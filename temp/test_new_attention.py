from src.models import BertModel
import jax
import equinox as eqx
import jax.numpy as jnp

from transformers import BertConfig, BertTokenizer
config = BertConfig(
    hidden_size= 64,
    num_attention_heads= 8,
    intermediate_size= 256,
    vocab_size= 30522,
    max_position_embeddings= 96,
    type_vocab_size= 2,
    layer_norm_eps= 1e-12,
    hidden_dropout_prob= 0.0,
    attention_probs_dropout_prob= 0.0,
    num_hidden_layers= 2,
)

model = BertModel(config, key = jax.random.key(10))
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
token = tokenizer("hallo saya makan nasi goreng", return_tensors = "np")
print(f"DEBUGPRINT[323]: test_new_attention.py:20: token={token}")
token["position_ids"] = jnp.arange(token["input_ids"].shape[-1])[None, :]

model = eqx.nn.inference_mode(model)
print(f"DEBUGPRINT[325]: test_new_attention.py:26: model={model}")
out = model(**token)

print(f"DEBUGPRINT[324]: test_new_attention.py:24: out={out}")
