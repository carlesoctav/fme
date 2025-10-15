import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from transformers import BertConfig
from src.models.bert import BertForMaskedLM
from src import DArray
import jax.tree_util as jtu
import os

os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text'

SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 2

def unbox_params(module):
    is_darray = lambda x: isinstance(x, DArray)
    def unbox(leaf):
        if isinstance(leaf, DArray):
            return leaf.value
        return leaf
    return jtu.tree_map(unbox, module, is_leaf=is_darray)

key = jr.PRNGKey(SEED)

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=1,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    _attn_implementation="eager",
)

model = BertForMaskedLM(config, key=key)
model = unbox_params(model)

input_ids = jnp.ones((BATCH_SIZE, MAX_LENGTH), dtype=jnp.int32)
position_ids = jnp.broadcast_to(jnp.arange(MAX_LENGTH)[None, :], (BATCH_SIZE, MAX_LENGTH))
token_type_ids = jnp.zeros((BATCH_SIZE, MAX_LENGTH), dtype=jnp.int32)
attention_mask = jnp.ones((BATCH_SIZE, MAX_LENGTH), dtype=jnp.int32)

@jax.jit
def forward(model, input_ids, position_ids, token_type_ids, attention_mask, key):
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        segment_ids=None,
        key=key,
    )

key, subkey = jr.split(key)

print("Running forward pass to trigger compilation...")
output = forward(model, input_ids, position_ids, token_type_ids, attention_mask, subkey)
output.block_until_ready()

print(f"\nHLO dumps saved to: /tmp/xla_dump")
print("\nLook for patterns like:")
print("  - fusion with 'dropout' operations")
print("  - fusion with 'attention_mask' or 'add' operations (bias mask)")
print("  - Check if dropout's random generation, multiply, and compare are fused")
