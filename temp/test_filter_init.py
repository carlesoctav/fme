import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import partitionspec as p
from jax.experimental.shard_map import shard_map
from typing import any, callable
from typing import Callable

from experiment.bert.train import filter_shard_map, annotate_params
from src.distributed._utils import simulate_cpu_devices

simulate_cpu_devices()


class tiny(eqx.module):
    w: jax.array

    def __init__(self, *, key):
        self.w = jax.random.normal(key, (4,))

    def __call__(self, *, key):
        return self.w


def main():
    mesh = jax.make_mesh((8,), ("data",), devices=jax.devices())

    # abstract module for specs
    abstract = eqx.filter_eval_shape(tiny, key=jax.random.key(0))
    pspec = annotate_params(abstract)
    print(f"debugprint[175]: test_filter_init.py:27: pspec={pspec}")

    def init_fn():
        return tiny(key=jax.random.key(0))

    wrapper = filter_shard_map(
        f=init_fn,
        mesh=mesh,
        in_specs=(),
        out_specs=pspec,
        check_rep=False,
    )

    m = wrapper()
    print("ok", type(m), m.w.shape)


if __name__ == "__main__":
    main()

