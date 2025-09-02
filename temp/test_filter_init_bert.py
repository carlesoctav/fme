import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P
from typing import Any

from experiment.bert.train import filter_shard_map, annotate_params
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert.modeling_bert import BertModel
from src.distributed._utils import simulate_CPU_devices

simulate_CPU_devices()


def main():
    mesh = jax.make_mesh((8,), ("data",), devices=jax.devices())
    cfg = BertConfig(num_hidden_layers=1)

    abstract = eqx.filter_eval_shape(BertModel, cfg, key=jax.random.key(0))
    pspec = annotate_params(abstract)

    def init_fn():
        return BertModel(cfg, key=jax.random.key(0))

    wrapper = filter_shard_map(
        f=init_fn,
        mesh=mesh,
        in_specs=(),
        out_specs=pspec,
        check_rep=False,
    )

    m = wrapper()
    print("BERT OK ->", type(m))


if __name__ == "__main__":
    main()

