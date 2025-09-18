import equinox as eqx
import jax
import jax.tree_util as jtu
from equinox import nn
from src.models.bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from src import iter_module



class SuperLinear(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    random_arr: jax.Array

    def __init__(self, params, key):
        self.linear1 = nn.Linear(params[0], params[1], key = key)
        self.linear2 = nn.Linear(params[1], params[2], key = key)
        self.random_arr = jax.random.normal(key, (1, 1))

cfg = BertConfig(
    voca_size = 30522,
    hidden_size = 64,
    num_hidden_layers = 2,
    num_attention_heads = 8,
    intermediate_size = 256,
    max_position_embeddings = 96,
    type_vocab_size = 2,
    layer_norm_eps = 1e-12,
    hidden_dropout_prob = 0.0,
    attention_probs_dropout_prob = 0.0,
)  # default config

model = BertModel(config = cfg, key = jax.random.key(1))

def visit(path, module):
    if isinstance(module, eqx.Module):
        print(jtu.keystr(path))



for path, sub_module in iter_module(model):
    print(f"DEBUGPRINT[307]: test_iter.py:42: path={path}")

