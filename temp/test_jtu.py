import jax.numpy as jnp
from typing import  Callable
import numpy as np
import equinox as eqx
import grain
import jax
from datasets import load_dataset
from grain._src.python.samplers import IndexSampler
from grain.transforms import Batch
from transformers import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from src.data import DataTransformsMakeAttentionMask
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert.modeling_bert import BertForMaskedLM
from optax import GradientTransformationExtraArgs
from jaxtyping import PyTree, Array
import optax
from src.distributed import get_dp_partition_spec
from jax import tree_util as jtu

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=128,
    type_vocab_size=2,
    hidden_dropout_prob=0.0,
    layer_norm_eps=1e-5,
)

key = jax.random.PRNGKey(0)
model = BertForMaskedLM(config, dtype=jnp.float32, key=key, store_config=True)

def _f(path, x):
    print(path)

jtu.tree_map_with_path(_f, model)
