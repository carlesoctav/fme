from src.models.bert import BertModel
from transformers import BertConfig
import jax


config = BertConfig(
    hidden_size = 64,
    num_attention_heads = 8,
    intermediate_size = 256,
    vocab_size = 30522,
    max_position_embeddings = 96,
    type_vocab_size = 2,
    layer_norm_eps = 1e-12,
    hidden_dropout_prob = 0.0,
    attention_probs_dropout_prob = 0.0,
    num_hidden_layers = 2,
)


print(f"DEBUGPRINT[309]: test_hf.py:20: BertModel.hf_model_class={BertModel.hf_model_class}")
model = BertModel.from_huggingface("google-bert/bert-base-uncased", key = jax.random.key(10))
print(f"DEBUGPRINT[308]: test_hf.py:20: model={model}")
