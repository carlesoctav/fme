import functools as ft
from collections.abc import Callable

import equinox as eqx
from src.distributed._utils import simulate_CPU_devices
import grain
import jax
from jax import P
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from grain._src.python.samplers import IndexSampler
from grain.transforms import Batch
from jaxtyping import Array, PyTree
from optax import GradientTransformationExtraArgs
from transformers import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from src.data import DataTransformsMakeAttentionMask
from src.distributed import get_dp_partition_spec
from src.models.bert.modeling_bert import BertForMaskedLM

simulate_CPU_devices()
devices = jax.devices()
print(f"DEBUGPRINT[236]: test_data_but_dp.py:113: devices={devices}")
mesh = jax.make_mesh((8,), ("data",), devices=jax.devices())


ds = load_dataset("carlesoctav/en-id-parallel-sentences", split = "QED")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
transformations = [
    DataTransformsMakeAttentionMask(tokenizer, columns = "text_en", max_length = 20),
    Batch(batch_size=16, drop_remainder = True)
]

# datasets = (
#     grain.MapDataset.source(ds)
#     .random_map(DataTransformsForMaskedLanguageModeling(tokenizer, columns = "text_en", max_length = 32), seed = 43)
#     .batch(2)
# )
#
datasets = grain.DataLoader(
    data_source = ds,
    operations = transformations, 
    sampler = IndexSampler(num_records = len(ds), shuffle = True, seed = 42)
)

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
# model = eqx.nn.inference_mode(model)
columns = "text_en"

def masked_lm_loss_function(
    model: BertForMaskedLM,
    batch,
    *,
    ignore_index: int = -100,
    key,
):
    """Compute masked LM loss, ignoring positions with label == ignore_index.

    - Vectorizes the model over batch.
    - Uses safe labels for ignored positions to avoid invalid indexing.
    - Normalizes by the number of unmasked tokens.
    """
    x, y = batch[columns], batch["labels"]  
    batch_size, seq_len = x["input_ids"].shape

    batched_key = jax.random.split(key, batch_size)
    logits = jax.vmap(model)(**x, key = batched_key)

    label_mask = jnp.where(y != ignore_index, 1.0, 0.0)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y) * label_mask # (..., ) (..., )

    return loss.sum()/ label_mask.sum() 

grad_fn = eqx.filter_value_and_grad(masked_lm_loss_function, has_aux=False)


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: Callable[[ PyTree[Array] ], bool] = eqx.field(static=True)
    step: int 
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, optimizer, model, *, wrt=eqx.is_inexact_array):
        self.tx = optimizer
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = 0

    def __call__(self, grads: PyTree[Array], model: eqx.Module) -> tuple[eqx.Module, "Optimizer"]:
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(model, self.wrt)
        )
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: [o.opt_state, o.step], self, [opt_state, self.step+1])
        return new_model, new_self



key, train_key = jax.random.split(key)
tx = optax.chain(
    # optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4, eps=1e-8),
)
optimizer = Optimizer(tx, model)
dp_partition_spec = get_dp_partition_spec(model) 

@ft.partial(
    jax.shard_map,
    in_specs=(P(), P("data"), P(), P()),
    out_specs=(P(), P(), P()),
    mesh=mesh,
)
@eqx.filter_jit
def train_step(model, batch, optimizer: Optimizer, key):
    print(f"DEBUGPRINT[237]: test_data_but_dp.py:130: model={model}")
    loss, grads = grad_fn(model, batch, key=key)
    model, optimizer = optimizer(grads, model)
    return model, optimizer, loss



for i, batch in enumerate(datasets):
    print(f"DEBUGPRINT[215]: test_data.py:115: i={i}")
    batch_size, seq_len = batch[columns]["input_ids"].shape
    train_key, subkey = jax.random.split(train_key)

    batch[columns]["position_ids"] = np.broadcast_to(
        np.arange(seq_len, dtype=np.int32), (batch_size, seq_len)
    )
    model, optimizer, loss = train_step(model, batch, optimizer, key=subkey)
    print(f"iter={i} loss={loss}")
    if i > 3:
        break
       
